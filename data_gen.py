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


def generate_ising_params(d, sigma_h=0.1, sigma_J=0.1, seed=0):

    key = jax.random.PRNGKey(seed)
    key_h, key_J = jax.random.split(key)

    # field
    h = sigma_h * jax.random.normal(key_h, shape=(d,))

    a = sigma_J * jax.random.normal(key_J, shape=(d, d))
    J = a.T + a

    # remove diagonal (optional for standard Ising)
    J = J - jnp.diag(jnp.diag(J))

    return h, J


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def energy_ising(sigma, h, J):
    return -jnp.dot(h, sigma) - 0.5 * jnp.dot(sigma, J @ sigma)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def glauber_step(key, state, h, J, beta):
    d = state.shape[0]

    key_i, key_flip = jax.random.split(key)

    i = jax.random.randint(key_i, shape=(), minval=0, maxval=d)

    interaction_term = jnp.dot(J[i], state) - J[i, i] * state[i]
    local_field = h[i] + interaction_term

    p_plus = 1.0 / (1.0 + jnp.exp(-2.0 * beta * local_field))

    flip = jax.random.bernoulli(key_flip, p_plus)
    new_spin = jnp.where(flip, 1, -1)  # choose +1 if flip is True, otherwise -1

    new_state = state.at[i].set(new_spin)

    return new_state


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def apply_glauber_to_all(keys, samples, h, J, beta):
    return jax.vmap(glauber_step, in_axes=(0, 0, None, None, None))(keys, samples, h, J, beta)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def compute_empirical_magnetization(samples):
    return jnp.mean(samples, axis=0)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def compute_self_consistent_magnetization(h, J, m_empirical, beta):
    return jnp.tanh(beta * (h + J @ m_empirical))


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


def generate_ising_data(n_samples, h, J, n_steps, beta, seed):
    d = h.shape[0]
    key_init = jax.random.PRNGKey(seed)
    key_samples, key_steps = jax.random.split(key_init)

    samples = initialize_samples(n_samples, d, key_samples)
    keys_all = generate_all_keys(seed + 1, n_steps, n_samples)

    for t in range(n_steps):
        keys_t = keys_all[t]
        samples = apply_glauber_to_all(keys_t, samples, h, J, beta)
        if t % 500 == 0:
            m_emp = compute_empirical_magnetization(samples)
            m_pred = compute_self_consistent_magnetization(h, J, m_emp, beta)
            print_magnetizations(t, m_emp, m_pred)

    return samples


# ----------------------------------------------------------------------------------------------------------
# MULTIVARIATE GAUSSIAN DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


def generate_gaussian_params(d, sigma_mu=0.1, sigma_cov=0.1, seed=0):
    """
    generate a random mean vector mu and a positive definite covariance matrix cov.

    args:
        d: dimension
        sigma_mu: stddev of the mean vector
        sigma_cov: scale of the covariance matrix
        seed: RNG seed

    returns:
        mu: vector of shape (d,)
        cov: positive definite covariance matrix (d, d)
    """
    key = jax.random.PRNGKey(seed)
    key_mu, key_cov = jax.random.split(key)

    mu = sigma_mu * jax.random.normal(key_mu, shape=(d,))

    # create random positive definite matrix: cov = A @ A.T + epsilon * I
    A = sigma_cov * jax.random.normal(key_cov, shape=(d, d))
    cov = A @ A.T + 1e-2 * jnp.eye(d)

    return mu, cov


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def generate_gaussian_data(mu, cov, n_samples, seed):
    """
    generate samples from a multivariate Gaussian N(mu, cov)

    args:
        mu: mean vector of shape (d,)
        cov: covariance matrix (d, d)
        n_samples: number of samples to generate
        seed: RNG seed

    returns:
        samples: array of shape (n_samples, d)
    """
    key = jax.random.PRNGKey(seed)

    # use Cholesky decomposition: cov = L @ L.T
    L = jnp.linalg.cholesky(cov)
    d = mu.shape[0]

    z = jax.random.normal(key, shape=(n_samples, d))
    samples = mu + z @ L.T  # z is (n_samples, d), L.T is (d, d)

    return samples