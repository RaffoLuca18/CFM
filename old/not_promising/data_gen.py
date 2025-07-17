# data_gen.py


# ----------------------------------------------------------------------------------------------------------
# LIBRARIES
# ----------------------------------------------------------------------------------------------------------


import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------
# ISING DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


def generate_ising_params(d, sigma_h=0.1, sigma_J=0.1, mean_h = 0, mean_J = 0, seed=0):

    key = jax.random.PRNGKey(seed)
    key_h, key_J = jax.random.split(key)

    h = sigma_h * jax.random.normal(key_h, shape=(d,)) + mean_h
    a = sigma_J * jax.random.normal(key_J, shape=(d, d)) + mean_J

    J = a.T + a
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

    interaction_term = jnp.dot(J[i], state)
    local_field = h[i] + interaction_term

    p_plus = 1.0 / (1.0 + jnp.exp(-2.0 * beta * local_field))

    flip = jax.random.bernoulli(key_flip, p_plus)
    new_spin = jnp.where(flip, 1, -1)
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


def plot_magnetization_evolution(m_empirical_list, m_predicted_list):

    m_empirical_array = jnp.stack(m_empirical_list)  # shape (T, d)
    m_predicted_array = jnp.stack(m_predicted_list)  # shape (T, d)

    T, d = m_empirical_array.shape
    x = jnp.arange(T)

    fig, axs = plt.subplots(d, 1, figsize=(8, 2.5 * d), sharex=True)

    for i in range(d):
        axs[i].plot(x, m_empirical_array[:, i], label="Empirical", linestyle="-")
        axs[i].plot(x, m_predicted_array[:, i], label="Predicted", linestyle="--")
        axs[i].set_ylabel(f"m[{i}]")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Glauber step")
    plt.tight_layout()
    plt.show()


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


def generate_ising_data(n_init, n_replicas, h, J, n_steps_equil, n_steps_final, n_prints, beta, seed):
    """
    1. evolves a few samples to equilibrium
    2. copies each to generate total of n_samples
    3. applies a few Glauber steps (n_steps_final) to decorrelate copies

    returns:
        samples of shape (n_samples, d)
    """
    d = h.shape[0]

    # 1. step: equilibrate n_init samples
    key_equil = jax.random.PRNGKey(seed)
    init_samples = initialize_samples(n_init, d, key_equil)
    keys_equil = generate_all_keys(seed + 1, n_steps_equil, n_init)

    for t in range(n_steps_equil + 1):

        init_samples = apply_glauber_to_all(keys_equil[t], init_samples, h, J, beta)

        if t == 0 or t % n_prints == 0:

            m_empirical = compute_empirical_magnetization(init_samples)
            print_magnetizations(t, m_empirical, compute_self_consistent_magnetization(h, J, m_empirical, beta))

    # 2. step: copy to get total n_samples
    reps = n_init * n_replicas
    samples = jnp.repeat(init_samples, repeats=n_replicas, axis=0)

    # 3. step: decorrelate copies
    n_samples = n_init * n_replicas
    keys_final = generate_all_keys(seed + 2, n_steps_final, n_samples)
    for t in range(n_steps_final):
        samples = apply_glauber_to_all(keys_final[t], samples, h, J, beta)

    return samples


# ----------------------------------------------------------------------------------------------------------
# MULTIVARIATE GAUSSIAN DATA GENERATION
# ----------------------------------------------------------------------------------------------------------


def generate_gaussian_params(d, sigma_mu=0.1, sigma_cov=0.1, seed=0):

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

    key = jax.random.PRNGKey(seed)

    # use Cholesky decomposition: cov = L @ L.T
    L = jnp.linalg.cholesky(cov)
    d = mu.shape[0]

    z = jax.random.normal(key, shape=(n_samples, d))
    samples = mu + z @ L.T

    return samples