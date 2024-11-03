import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
iterations = 1_000_000

for _ in range(iterations):
    matrix_a = torch.rand(4, 4, device=device)
    matrix_b = torch.rand(4, 4, device=device)
    result = torch.matmul(matrix_a, matrix_b)
    print(result)
