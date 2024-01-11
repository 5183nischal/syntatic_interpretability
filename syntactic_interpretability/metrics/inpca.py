from scipy.special import softmax
import numpy as np

# TODO: Add jaxtyping to this
def bhattacharyya_distance_matrix(mat, batch_size: int= 500):
    """
    Takes a collection of models and computes the pairwise
    Bhattacharyya distance between all pairs of models.
    """
    mat = np.sqrt(mat) + 1e-9
    mat1 = np.transpose(mat, axes=[1, 0, 2])
    mat2 = np.transpose(mat, axes=[1, 2, 0])

    dim = len(mat1)

    # Compute distance in batches to avoid OOM
    Dmat = 0.0
    for i in range(0, dim, batch_size):
        Dmat += (np.log(mat1[i:i+batch_size] @ mat2[i:i+batch_size])).sum(0)
    Dmat = Dmat / dim
    return Dmat

# TODO: This needs to have a new name, given the function below
def compute_inpca(Dmat):
    """
    Compute the InPCA embedding from a pairwise distance matrix
    """
    # Double center matrix
    ldim = Dmat.shape[0]
    Pmat = np.eye(ldim) - 1.0/ ldim
    Wmat = (Pmat @ Dmat @ Pmat) / 2

    eigenval, eigenvec = np.linalg.eigh(Wmat)

    #Sort eigen-values by magnitude
    sort_ind = np.argsort(-np.abs(eigenval))
    eigenval = eigenval[sort_ind]
    eigenvec = eigenvec[:, sort_ind]
    sqrt_eigenval = np.sqrt(np.abs(eigenval))

    # Find projections
    projection = eigenvec * sqrt_eigenval.reshape(1, -1)

    return eigenval, projection

# TODO: Rename this... I'm not sure exactly why we are doing all of this
def inpca(model_predictions):
  """
  This is an approach to visualize the trajectory of the network during training in functional space.
  
  The idea was first derived and discussed here:
  https://www.pnas.org/doi/10.1073/pnas.1817218116
  """

  predictions = model_predictions.to('cpu').detach().numpy() # Why??
  probabilities = softmax(predictions, axis=2) 
  distance_matrix = bhattacharyya_distance_matrix(probabilities)
  eigenval, embed = compute_inpca(distance_matrix)
  embed = embed.reshape([4, len(checkpoint_indices), -1]) # Why??
  return embed

# TODO: Add description
def in_context_learning_score(model, tokens) -> float:
    loss_vec = model(tokens, return_type='loss', loss_per_token=True)
    return (loss_vec[..., 500] - loss_vec[..., 50]).mean()