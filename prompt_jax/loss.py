import jax.numpy as jnp

# def cos_sim(x,y):
#     return jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y))

def mse(x,y):
    x = normalize(x,dim=-1,p=2)
    y = normalize(y,dim=-1,p=2)
    return 2 - 2 * (x * y).sum(axis=-1)

def normalize(x, dim=-1, p=2,ep=1e-6):
    norm = jnp.linalg.norm(x, axis=dim, keepdims=True,ord=p)
    denominator = jnp.maximum(norm, jnp.ones_like(norm) * ep)
    outputs = x / denominator 

    return outputs

