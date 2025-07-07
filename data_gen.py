# data_gen.py


# ----------------------------------------------------------------------------------------------------------
# LIBRARIES
# ----------------------------------------------------------------------------------------------------------


import numpy as np
import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------------------------------------------
# ISING DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


def generate_log_concave_ising_params(d, sigma_h=0.1, sigma_j=0.1, seed=0):
    """
    generate field h and interaction matrix j such that the Boltzmann distribution is log-concave
    (i.e., j is negative semidefinite)

    args:
        d: number of spins
        sigma_h: stddev of field
        sigma_j: scale of negative interaction
        seed: rng seed

    returns:
        h: vector of shape (d,)
        j: symmetric matrix (d, d), negative semidefinite with zero diagonal
    """
    key = jax.random.PRNGKey(seed)
    key_h, key_j = jax.random.split(key)

    # field
    h = sigma_h * jax.random.normal(key_h, shape=(d,))

    # make j = -A.T @ A, which is always negative semidefinite
    a = sigma_j * jax.random.normal(key_j, shape=(d, d))
    j = -a.T @ a  # symmetric, negative semidefinite

    # remove diagonal (optional for standard Ising)
    j = j - jnp.diag(jnp.diag(j))

    return h, j


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def energy_ising(sigma, h, j):
    return -jnp.dot(h, sigma) - 0.5 * jnp.dot(sigma, j @ sigma)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def glauber_step(key, state, h, j, beta):
    d = state.shape[0]

    key_i, key_flip = jax.random.split(key)

    i = jax.random.randint(key_i, shape=(), minval=0, maxval=d)

    interaction_term = jnp.dot(j[i], state) - j[i, i] * state[i]
    local_field = h[i] + interaction_term

    p_plus = 1.0 / (1.0 + jnp.exp(-2.0 * beta * local_field))

    flip = jax.random.bernoulli(key_flip, p_plus)
    new_spin = jnp.where(flip, 1, -1)  # choose +1 if flip is True, otherwise -1

    new_state = state.at[i].set(new_spin)

    return new_state



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def apply_glauber_to_all(samples, keys, h, j, beta):
    return jax.vmap(glauber_step, in_axes=(0, 0, None, None, None))(keys, samples, h, j, beta)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def compute_empirical_magnetization(samples):
    return jnp.mean(samples, axis=0)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def compute_self_consistent_magnetization(h, j, m_empirical, beta):
    return jnp.tanh(beta * (h + j @ m_empirical))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def print_magnetizations(t, m_empirical, m_predicted):
    lhs = jnp.round(m_empirical, 3)
    rhs = jnp.round(m_predicted, 3)
    print(f"step {t}:{lhs.tolist()}\n{rhs.tolist()}\n{jnp.abs(jnp.mean(lhs - rhs))}\n")



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def initialize_samples(n_samples, d, key):
    return jax.random.choice(key, jnp.array([-1, 1]), shape=(n_samples, d))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def generate_all_keys(seed, n_steps, n_samples):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_steps * n_samples)
    return keys.reshape((n_steps, n_samples, 2))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def generate_ising_data(n_samples, h, j, n_steps, beta, seed):
    d = h.shape[0]
    key_init = jax.random.PRNGKey(seed)
    key_samples, key_steps = jax.random.split(key_init)

    samples = initialize_samples(n_samples, d, key_samples)
    keys_all = generate_all_keys(seed + 1, n_steps, n_samples)

    for t in range(n_steps):
        keys_t = keys_all[t]
        samples = apply_glauber_to_all(samples, keys_t, h, j, beta)
        if t % 500 == 0:
            m_emp = compute_empirical_magnetization(samples)
            m_pred = compute_self_consistent_magnetization(h, j, m_emp, beta)
            print_magnetizations(t, m_emp, m_pred)

    return samples


# ----------------------------------------------------------------------------------------------------------
# GAUSSIAN DATA GENERATION
# ----------------------------------------------------------------------------------------------------------

