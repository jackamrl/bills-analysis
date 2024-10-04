import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from scipy.ndimage import gaussian_filter

# Parcourir les dossiers et fichiers JSON
def load_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                json_data.append(json.load(f))
    return json_data

# Extraction des informations des fichiers JSON
def analyze_json_data(json_data):
    total_count = 0
    subtotal_count = 0
    tax_images = 0
    entite_counts = {}
    total_positions = []
    tax_counts_per_image = []  

    for data in json_data:
        tax_count = 0  
        
        for line in data['valid_line']:
            category = line['category']
            
            # Compter les occurrences de Total et Sous-total
            if category == 'total.total_price':
                total_count += 1
                # Enregistrer les positions des montants totaux
                for word in line['words']:
                    if word['text'] == 'TOTAL':
                        total_positions.append((word['quad']['x1'], word['quad']['y1'], word['quad']['x2'], word['quad']['y3']))
            elif category == 'sub_total.subtotal_price':
                subtotal_count += 1

            if category == 'sub_total.tax_price':
                tax_count += 1

            if category not in entite_counts:
                entite_counts[category] = 0
            entite_counts[category] += 1

        tax_counts_per_image.append(tax_count)

    return total_count, subtotal_count, tax_images, entite_counts, total_positions, tax_counts_per_image

# Utilisation de la fonction pour charger les données et faire l'analyse
train_dir = "your_json_folder"
train_data = load_json_files(train_dir)

total_count, subtotal_count, tax_images, entite_counts, total_positions, tax_counts_per_image = analyze_json_data(train_data)


# --------------------------------------------------------------------------------------------------------------

# *** Question 1 - Vérification des positions des montants totaux ***

# *** Camembert pour Total et Sous-total ***
labels = ['Total', 'Sous-total']
sizes = [total_count, subtotal_count]
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution des Totaux et Sous-totaux')
plt.axis('equal')  # Pour un camembert circulaire
plt.show()

# --------------------------------------------------------------------------------------------------------------

# *** Question 2 - Distribution des taxes par nombre d'images ***
tax_count_distribution = Counter(tax_counts_per_image)

# Créer un DataFrame pour la distribution des taxes
tax_distribution_df = pd.DataFrame(tax_count_distribution.items(), columns=['Nombre de Taxes', 'Nombre d\'Images'])
tax_distribution_df.sort_values(by='Nombre de Taxes', inplace=True)

# Produire un graphique pour le nombre d'images par nombre de taxes (avec Y comme nombre de taxes et X comme nombre d'images)
plt.figure(figsize=(10, 6))
plt.bar(tax_distribution_df['Nombre d\'Images'], tax_distribution_df['Nombre de Taxes'])
plt.title('Distribution du Nombre de Taxes par Nombre d\'Images')
plt.xlabel('Nombre d\'Images')
plt.ylabel('Total des Taxes')
plt.xticks(tax_distribution_df['Nombre d\'Images'])  # Étiquettes sur l'axe X
min_y = tax_distribution_df['Nombre de Taxes'].min()
max_y = tax_distribution_df['Nombre de Taxes'].max()
plt.yticks(range(min_y, max_y + 1))
plt.show()

# Compter le nombre d'images ayant plus de 1 taxe
images_with_multiple_taxes = sum(1 for count in tax_counts_per_image if count > 1)
print(f"Nombre d'images ayant plusieurs taxes : {images_with_multiple_taxes}")


# --------------------------------------------------------------------------------------------------------------

# *** Question 3 - HEATMAP pour les positions des montants totaux ***

# *** HEATMAP pour les positions des montants totaux ***
heatmap_data = np.zeros((1296, 864))  # Dimensions de l'image (hauteur, largeur)

# Remplir la heatmap avec les positions des montants totaux
for pos in total_positions:
    x_center = int((pos[0] + pos[2]) / 2)
    y_center = int((pos[1] + pos[3]) / 2)
    if 0 <= x_center < 864 and 0 <= y_center < 1296:
        heatmap_data[y_center, x_center] += 1

# Option 1 : Application d'un filtre gaussien pour lisser la heatmap
heatmap_data_smoothed = gaussian_filter(heatmap_data, sigma=10)  # Ajustez sigma pour plus ou moins de lissage

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data_smoothed, cmap="YlGnBu", cbar_kws={'label': 'Fréquence des montants totaux'})
plt.title('Heatmap des positions des Montants Totaux (avec lissage)')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.gca().invert_yaxis()

plt.show()

# --------------------------------------------------------------------------------------------------------------
# *** Question 4 - Création d'un graphique pour les entités et identification de l'entité la moins représentée ***
if entite_counts:
    entite_df = pd.DataFrame.from_dict(entite_counts, orient='index', columns=['Occurrences'])
    
    # Trouver l'entité la moins représentée
    least_represented_entity = entite_df['Occurrences'].idxmin()
    least_represented_count = entite_df['Occurrences'].min()
    
    print(f"L'entité la moins représentée est '{least_represented_entity}' avec {least_represented_count} occurrences.")
    
    # Tracer un graphique pour toutes les entités
    plt.figure(figsize=(10, 6))
    entite_df.sort_values(by='Occurrences', inplace=True)  # Trier par ordre croissant
    entite_df.plot(kind='bar', legend=False)
    plt.title("Occurrence des Entités")
    plt.xlabel("Entité")
    plt.ylabel("Nombre d'Occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Aucune entité trouvée pour l'analyse.")


