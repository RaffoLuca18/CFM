# functs.py


# ----------------------------------------------------------------------------------------------------------
# LIBRARIES
# ----------------------------------------------------------------------------------------------------------


import numpy as np
import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------------------------------------------
# ISING DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


def energy_ising(sigma, h, j):
    return -jnp.dot(h, sigma) - 0.5 * jnp.dot(sigma, j @ sigma)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def glauber_step(key, state, h, j, beta):
    d = state.shape[0]
    key_i, key_flip = jax.random.split(key)
    i = jax.random.randint(key_i, shape=(), minval=0, maxval=d)
    local_field = h[i] + jnp.dot(j[i], state) - j[i, i] * state[i]
    p = 1 / (1 + jnp.exp(-2 * beta * local_field))
    new_spin = jnp.where(jax.random.bernoulli(key_flip, p), 1, -1)
    return state.at[i].set(new_spin)


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
# GAUSSIAN DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------
# DISCRETE MPF
# ----------------------------------------------------------------------------------------------------------


def flip_spin(sigma, i):
    return sigma.at[i].set(-sigma[i])


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def local_energy_diff(sigma, i, h, j):
    """
    Compute energy difference E(new_sigma) - E(sigma) when flipping spin i
    """
    delta = 2 * sigma[i] * (h[i] + jnp.dot(j[i], sigma) - j[i, i] * sigma[i])
    return delta


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def mpf_loss_single_sample(sigma, h, j, beta=1.0):
    """
    Compute the MPF loss for a single sample sigma
    """
    d = sigma.shape[0]
    def energy_diff(i):
        delta_e = local_energy_diff(sigma, i, h, j)
        return jnp.exp(-0.5 * beta * delta_e)
    
    return jnp.sum(jax.vmap(energy_diff)(jnp.arange(d)))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def compute_energy_differences_all_sites(sigma, h, j):
    """
    Return vector of energy differences for all single-spin flips of one configuration
    """
    d = sigma.shape[0]
    indices = jnp.arange(d)

    def single_diff(i):
        return local_energy_diff(sigma, i, h, j)

    return jax.vmap(single_diff)(indices)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def mpf_loss_per_sample(sigma, h, j, beta):
    """
    MPF loss for one sample (no inner function, no lambda)
    """
    delta_e = compute_energy_differences_all_sites(sigma, h, j)
    return jnp.sum(jnp.exp(-0.5 * beta * delta_e))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def mpf_loss(samples, h, j, beta=1.0):
    """
    Average MPF loss over all samples
    """
    return jnp.mean(jax.vmap(mpf_loss_per_sample, in_axes=(0, None, None, None))(samples, h, j, beta))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def mpf_gradients(samples, h, j, beta=1.0):
    """
    Compute gradients of MPF loss w.r.t. h and j
    """
    grad_loss = jax.grad(mpf_loss, argnums=(1, 2))
    return grad_loss(samples, h, j, beta)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def optimize_mpf(samples, h_init, j_init, n_steps=1000, lr=1e-2, beta=1.0):
    """
    Perform simple gradient descent on MPF loss with symmetrization of j
    """
    h = h_init
    j = j_init

    for t in range(n_steps):
        grad_h, grad_j = mpf_gradients(samples, h, j, beta)

        h = h - lr * grad_h
        j = j - lr * grad_j

        # Enforce symmetry and zero diagonal on J
        j = 0.5 * (j + j.T)
        j = j - jnp.diag(jnp.diag(j))

        if t % 100 == 0:
            loss_val = mpf_loss(samples, h, j, beta)
            print(f"step {t} | loss = {loss_val:.6f}")

    return h, j
