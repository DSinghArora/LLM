import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class TeacherModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x):
        x= self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x #logits
    
class StudentModel(nn.Module):
     def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

     def forward(self, x):
        x= self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


input_dim = 100  # Size of our "text embedding"
output_dim = 5   # Number of classes (e.g., categories of inquiries)
teacher_hidden_dim = 256
student_hidden_dim = 64 # Student is smaller

num_samples = 1000
batch_size=64
num_epochs_teacher = 100
num_epochs_student = 200
learning_rate_teacher = 0.01
learning_rate_student = 0.005

temperature = 2.0
alpha = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X = torch.randn(num_samples,input_dim).to(device)
Y = torch.randint(0, output_dim, size=(num_samples,)).to(device)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


teacher_model = TeacherModel(input_dim, teacher_hidden_dim, output_dim)
student_model = StudentModel(input_dim, student_hidden_dim, output_dim)
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate_teacher)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate_student)
criterion_ce = nn.CrossEntropyLoss() # For hard targets
criterion_kl = nn.KLDivLoss(reduction='batchmean') # For soft targets (KL divergence)


teacher_model.to(device)
teacher_model.train()
for epoch in range(num_epochs_teacher):
    total_loss = 0
    correct_predictions = 0
    for inputs, labels in dataloader:
         inputs, labels = inputs.to(device), labels.to(device)
         teacher_optimizer.zero_grad()
         outputs = teacher_model(inputs)
         loss = criterion_ce(outputs, labels)
         teacher_optimizer.step()
         loss.backward()
         total_loss += loss.item()
         _, predicted = torch.max(outputs.data, 1)
         correct_predictions += (predicted == labels).sum().item()
         avg_loss = total_loss / len(dataloader)
         accuracy = correct_predictions / (len(dataloader) * batch_size) * 100
         if (epoch + 1) % 10 == 0:
            print(f'Teacher Epoch [{epoch+1}/{num_epochs_teacher}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')



#DiStill
teacher_model.eval() # Set teacher to evaluation mode (no gradient updates)
student_model.to(device)
student_model.train() # Set student to training mode


for epoch in range(num_epochs_student):
    total_loss = 0
    correct_predictions = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        student_optimizer.zero_grad()

        # Teacher's output (soft targets)
        with torch.no_grad(): # No gradient calculation for teacher
            teacher_logits = teacher_model(inputs)
            # Apply temperature to teacher's logits and then softmax
             # log_softmax is more numerically stable with KLDivLoss
            teacher_probs = nn.functional.log_softmax(teacher_logits / temperature, dim=1)
        

         # Student's output
        student_logits = student_model(inputs)
        # Apply temperature to student's logits and then softmax
        student_log_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)

        loss_kd = criterion_kl(student_log_probs, teacher_probs)* temperature**2 #scaled by T^2
        loss_ce = criterion_ce(student_logits, labels)
        # Combine losses
        loss = (alpha * loss_kd) + ((1 - alpha) * loss_ce)
        loss.backward()
        student_optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(student_logits.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / (len(dataloader) * batch_size) * 100
        if (epoch + 1) % 10 == 0:
            print(f'Student Epoch [{epoch+1}/{num_epochs_student}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')


def evaluate_model(model, dataloader, model_name):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    accuracy = correct_predictions / total_samples * 100
    print(f'{model_name} Accuracy: {accuracy:.2f}%')
    return accuracy

teacher_acc = evaluate_model(teacher_model, dataloader, "Teacher Model")
student_acc = evaluate_model(student_model, dataloader, "Student Model")



