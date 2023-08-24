import torch

# Create a tensor of ones 
a = torch.ones(3)
print("a * 2 = ", a * 2)
print("a + 5 = ", a + 5)
print("a - 1 = ", a - 1)

print(" 3 % 2 = ", 3 % 2)
print("43 % 8 = ", 43 % 8)

print("int(3.99) = ", int(3.99))

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

print("x * y = ", x * y)
print("torch.concantenate((x, y), dim=0) = ", torch.cat((x, y), dim=0))

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
print("x * y = ", x * y)
print("torch.concantenate((x, y), dim=0) = ", torch.cat((x, y), dim=0))


print("torch.cat((x, y), dim=0) = ", torch.cat((x, y), dim=0))