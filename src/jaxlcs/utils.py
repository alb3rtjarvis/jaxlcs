import numpy as np
import jax.numpy as jnp
# from jaxtyping import Float, Array

def neighbor_veiws(arr, mode='constant'):
    
    pad_width = ((0, 0) * (arr.ndim - 2), (1, 1) * 2)
    
    pad_arr = jnp.pad(arr, pad_width, mode=mode)
    
    nhbr_views = {
        'center': pad_arr[..., 1:-1, 1:-1],
        'west': pad_arr[..., :-2, 1:-1],
        'east': pad_arr[..., 2:, 1:-1],
        'north': pad_arr[..., 1:-1, :-2],
        'south': pad_arr[..., 1:-1, 2:],
        'nw': pad_arr[..., :-2, :-2],
        'ne': pad_arr[..., 2:, :-2],
        'sw': pad_arr[..., :-2, 2:],
        'se': pad_arr[..., 2:, 2:]
    }
    
    return nhbr_views
        
    
    
    


def collapse_leading_dims(arr, k):
    """
    

    Parameters
    ----------
    arr : Array
        array that will be reshaped.
    k : int
        number of leading dimensions to collapse.

    Returns
    -------
    Array
        reshaped array

    """
    
    shape = arr.shape
    
    return arr.reshape((np.prod(shape[:k]), ) + shape[k:])