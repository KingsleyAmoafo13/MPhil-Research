# #Camparing with base on PASCAL
import matplotlib.pyplot as plt

# # Data from the provided table
# methods = ['SCCNet', 'Proposed method']
# folds = ['Fold0', 'Fold1', 'Fold2']
# sccnet_scores = [32.22, 52.40, 47.17]
# proposed_scores = [45.35, 53.33, 49.99]

# # Plotting the line graph
# plt.figure(figsize=(8, 6))
# plt.plot(folds, sccnet_scores, marker='o', label='SCCNet', linestyle='-', color='blue')
# plt.plot(folds, proposed_scores, marker='o', label='Proposed Method', linestyle='-', color='red')

# # Adding labels and title
# plt.xlabel('Folds')
# plt.ylabel('mIoU')

# plt.legend()
# #plt.title('Performance Comparison: SCCNet vs Proposed Method')
# # Display the plot
# plt.grid(True)
# plt.show()


#baseline Comparison for ISAID

# Data for the new table with three backbones
# folds = ['Fold 0', 'Fold 1', 'Fold 2']

# # SCCNet and Proposed method for each backbone
# # VGG-16 Backbone
# sccnet_vgg16_scores = [30.00, 27.41, 32.43]
# proposed_vgg16_scores = [39.08, 28.44, 38.58]

# # ResNet-50 Backbone
# sccnet_resnet50_scores = [36.21, 27.42, 43.37]
# proposed_resnet50_scores = [37.92, 28.31, 44.94]

# # ResNet-101 Backbone
# sccnet_resnet101_scores = [37.65, 29.19, 42.99]
# proposed_resnet101_scores = [45.81, 30.06, 48.26]

# # Creating plots for each backbone
# fig, ax = plt.subplots(3, 1, figsize=(6, 12))

# # Plot for VGG-16
# ax[0].plot(folds, sccnet_vgg16_scores, marker='o', label='SCCNet', linestyle='-', color='blue')
# ax[0].plot(folds, proposed_vgg16_scores, marker='o', label='Proposed Method', linestyle='-', color='red')
# #ax[0].set_title('Performance Comparison: SCCNet vs Proposed Method (VGG-16)')
# ax[0].set_xlabel('Folds')
# ax[0].set_ylabel('mIoU')
# ax[0].legend()
# ax[0].grid(True)

# # Plot for ResNet-50
# ax[1].plot(folds, sccnet_resnet50_scores, marker='o', label='SCCNet', linestyle='-', color='blue')
# ax[1].plot(folds, proposed_resnet50_scores, marker='o', label='Proposed Method', linestyle='-', color='red')
# #ax[1].set_title('Performance Comparison: SCCNet vs Proposed Method (ResNet-50)')
# ax[1].set_xlabel('Folds')
# ax[1].set_ylabel('mIoU')
# ax[1].legend()
# ax[1].grid(True)

# # Plot for ResNet-101
# ax[2].plot(folds, sccnet_resnet101_scores, marker='o', label='SCCNet', linestyle='-', color='blue')
# ax[2].plot(folds, proposed_resnet101_scores, marker='o', label='Proposed Method', linestyle='-', color='red')
# #ax[2].set_title('Performance Comparison: SCCNet vs Proposed Method (ResNet-101)')
# ax[2].set_xlabel('Folds')
# ax[2].set_ylabel('mIoU')
# ax[2].legend()
# ax[2].grid(True)

# # Adjust layout and display the plots
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt

# # Data for the table with ResNet-50 Backbone
# folds = ['Fold 0', 'Fold 1', 'Fold 2']

# # Methods and their scores
# methods = ['DCAMA', 'PFENet', 'SDM', 'MLC', 'HSNet', 'PCFNet', 'SCCNet', 'MGCD', 'Proposed method']
# dcama_scores = [12.68, 13.01, 10.60]
# pfenet_scores = [21.98, 20.67, 11.17]
# sdm_scores = [20.11, 30.84, 27.87]
# mlc_scores = [26.38, 36.05, 34.65]
# hsnet_scores = [22.00, 47.20, 34.73]
# pcfnet_scores = [31.09, 37.52, 41.89]
# sccnet_mgcd_scores = [25.34, 48.97, 39.73]
# mgcd_scores = [30.37, 50.57, 43.01]
# proposed_scores = [43.58, 51.39, 44.80]



# # Creating the plot
# plt.figure(figsize=(10, 8))

# # Plot for each method
# plt.plot(folds, dcama_scores, marker='o', label='DCAMA', linestyle='-', color='violet')
# plt.plot(folds, pfenet_scores, marker='o', label='PFENet', linestyle='-', color='green')
# plt.plot(folds, sdm_scores, marker='o', label='SDM', linestyle='-', color='black')
# plt.plot(folds, mlc_scores, marker='o', label='MLC', linestyle='-', color='orange')
# plt.plot(folds, hsnet_scores, marker='o', label='HSNet', linestyle='-', color='purple')
# plt.plot(folds, pcfnet_scores, marker='o', label='PCFNet', linestyle='-', color='brown')
# plt.plot(folds, sccnet_mgcd_scores, marker='o', label='SCCNet', linestyle='-', color='pink')
# plt.plot(folds, mgcd_scores, marker='o', label='MGCD', linestyle='-', color='blue')
# plt.plot(folds, proposed_scores, marker='o', label='Proposed method', linestyle='-', color='red')

# # Adding labels and title
# #plt.xlabel('Folds')
# plt.ylabel('mIoU')
# #plt.title('Performance Comparison: Various Methods with ResNet-50 Backbone')
# plt.legend()

# # Display the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#SOTA ISAID vgg16

# import matplotlib.pyplot as plt

# # Data for the new table
# folds = ['Fold 0', 'Fold 1', 'Fold 2']

# # Methods and their scores
# methods = ['PANet', 'PFENet', 'PMMs', 'CANet', 'HSNet', 'SDM', 'SCCNet', 'MGCD', 'Proposed method']
# panet_scores = [17.43, 11.43, 15.95]
# pfenet_scores = [16.68, 15.30, 27.87]
# pmms_scores = [20.87, 16.07, 24.65]
# canet_scores = [19.73, 17.98, 30.93]
# hsnet_scores = [22.74, 23.05, 25.76]
# sdm_scores = [29.24, 20.80, 34.73]
# sccnet_scores = [30.00, 27.41, 32.43]
# mgcd_scores = [38.24, 28.61, 40.21]
# proposed_scores = [39.08, 28.44, 38.58]

# # Creating the plot
# plt.figure(figsize=(10, 8))

# # Plot for each method
# plt.plot(folds, panet_scores, marker='o', label='PANet', linestyle='-', color='cyan')
# plt.plot(folds, pfenet_scores, marker='o', label='PFENet', linestyle='-', color='green')
# plt.plot(folds, pmms_scores, marker='o', label='PMMs', linestyle='-', color='black')
# plt.plot(folds, canet_scores, marker='o', label='CANet', linestyle='-', color='orange')
# plt.plot(folds, hsnet_scores, marker='o', label='HSNet', linestyle='-', color='purple')
# plt.plot(folds, sdm_scores, marker='o', label='SDM', linestyle='-', color='brown')
# plt.plot(folds, sccnet_scores, marker='o', label='SCCNet', linestyle='-', color='pink')
# plt.plot(folds, mgcd_scores, marker='o', label='MGCD', linestyle='-', color='blue')
# plt.plot(folds, proposed_scores, marker='o', label='Proposed method', linestyle='-', color='red')

# # Adding labels and title
# #plt.xlabel('Folds')
# plt.ylabel('mIoU')
# #plt.title('Performance Comparison: Various Methods Across Folds')
# plt.legend()

# # Display the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# #SOTA RESNET50
# # Data for the table with one backbone
# folds = ['Fold 0', 'Fold 1', 'Fold 2']

# # Methods and their scores
# methods = ['PANet', 'PFENet', 'CANet', 'PMMs', 'SDM', 'HSNet', 'SCCNet', 'MGCD', 'Proposed method']
# panet_scores = [13.82, 12.4, 19.12]
# pfenet_scores = [19.57, 18.43, 26.14]
# canet_scores = [23.86, 18.54, 32.00]
# pmms_scores = [20.89, 20.87, 31.23]
# sdm_scores = [39.88, 30.59, 45.70]
# hsnet_scores = [38.08, 30.56, 45.28]
# sccnet_scores = [42.58, 30.30, 50.56]
# mgcd_scores = [49.14, 32.58, 52.75]
# proposed_scores = [46.68, 31.54, 53.49]

# # Creating the plot
# plt.figure(figsize=(10, 8))

# # Plot for each method
# plt.plot(folds, panet_scores, marker='o', label='PANet', linestyle='-', color='cyan')
# plt.plot(folds, pfenet_scores, marker='o', label='PFENet', linestyle='-', color='green')
# plt.plot(folds, canet_scores, marker='o', label='CANet', linestyle='-', color='black')
# plt.plot(folds, pmms_scores, marker='o', label='PMMs', linestyle='-', color='orange')
# plt.plot(folds, sdm_scores, marker='o', label='SDM', linestyle='-', color='purple')
# plt.plot(folds, hsnet_scores, marker='o', label='HSNet', linestyle='-', color='brown')
# plt.plot(folds, sccnet_scores, marker='o', label='SCCNet', linestyle='-', color='pink')
# plt.plot(folds, mgcd_scores, marker='o', label='MGCD', linestyle='-', color='blue')
# plt.plot(folds, proposed_scores, marker='o', label='Proposed method', linestyle='-', color='red')

# # Adding labels and title
# #plt.xlabel('Folds')
# plt.ylabel('mIoU')
# #plt.title('Performance Comparison: Various Methods Across Folds')
# plt.legend()

# # Display the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Data for the table with one backbone

#SOTA RESNET_101
folds = ['Fold 0', 'Fold 1', 'Fold 2']

# Methods and their scores
methods = ['HSNet', 'SCCNet', 'MGCD', 'Proposed method']
hsnet_scores = [41.71, 31.08, 48.54]
sccnet_scores = [41.87, 32.12, 49.63]
mgcd_scores = [50.58, 31.38, 54.02]
proposed_scores = [49.10, 33.34, 55.63]

# Creating the plot
plt.figure(figsize=(10, 8))

# Plot for each method
plt.plot(folds, hsnet_scores, marker='o', label='HSNet', linestyle='-', color='black')
plt.plot(folds, sccnet_scores, marker='o', label='SCCNet', linestyle='-', color='green')
plt.plot(folds, mgcd_scores, marker='o', label='MGCD', linestyle='-', color='blue')
plt.plot(folds, proposed_scores, marker='o', label='Proposed method', linestyle='-', color='red')

# Adding labels and title
#plt.xlabel('Folds')
plt.ylabel('mIoU')
#plt.title('Performance Comparison: Various Methods Across Folds')
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()

