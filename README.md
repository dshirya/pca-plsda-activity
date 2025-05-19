
# Activity Overview

This repository provides an educational activity for chemistry students and professionals to learn basic machine learning techniques – specifically PCA and PLS-DA – in the context of inorganic chemistry. The activity made in the **application** format, that can easily be deployed to the web to explore how elemental properties can be used as features for ML models. The pedagogical goal is to put a modern, data-driven spin on Linus Pauling’s 1929 study of binary compounds. In 1929, Pauling famously attempted to rationalize crystal structures of equiatomic binary compounds using simple rules based on atomic properties . This project revisits that idea with modern feature engineering and machine learning: we use a rich set of element-derived features to see if we can cluster and classify binary **AB** compounds (where A and B are elements) into their correct structure types (CsCl, NaCl, or ZnS). By combining Pauling’s early intuition with today’s ML tools, users will learn about **feature selection**, **unsupervised clustering vs. supervised classification**, and how to interpret ML results in chemical terms.

![app-overview](https://github.com/user-attachments/assets/007a1735-e4c0-4513-9830-6a78a315e5d9)

# Educational Goals and Methodology

This project is built around a **hands-on, visual, and interactive learning experience** to help chemistry students understand the foundations of machine learning (ML) in a chemical context — specifically **dimensionality reduction**, **feature selection**, and **classification** of crystal structures. The educational activity is split into two parts — PCA and PLS-DA — each with its own methodology and objectives.

---

### PCA as a New Perspective on the Periodic Table

In this activity, PCA (Principal Component Analysis) is used to construct a **data-driven periodic table**, leveraging a modern set of 74 numerical features — far more than were historically available. These include atomic, electronic, thermal, and DFT-derived properties.
![pca](https://github.com/user-attachments/assets/28c38145-27e0-45df-ac4c-97f473c4477e)



- Students use **interactive feature toggles** to include or exclude specific properties or whole categories.
- By doing so, they observe **how different features reshape the PCA projection** of elements and how elements cluster differently.
- This allows them to explore a mode
rn reinterpretation of the periodic table based not on empirical arrangement but on statistical similarities in high-dimensional property space.

Additionally, the PCA notebook includes a **second visualization**: compounds from Linus Pauling’s 1929 study are plotted onto the same PCA space, visualized as lines connecting constituent elements. The midpoint is marked, and compounds are colored by structure type (CsCl, NaCl, ZnS). This historical visualization allows students to explore:
- Whether structural clusters emerge from Pauling-era data
- How different subsets of properties impact separability
- Which features might have been intuitively used by Pauling, and how ML confirms or expands on those ideas
![clust](https://github.com/user-attachments/assets/99609e93-c292-4fe9-be48-f9bbe992cd31)


---

### PLS-DA and Visualization of Featurized Binary Compounds

The PLS-DA notebook works with the **same compounds** as the PCA activity, but uses a **supervised learning approach**. All compounds are labeled by their structure type, and 133 features are generated using the **CAF (Composition Analyzer/Featurizer)** tool.



- The features include **averages, differences, ratios, max/min** values across properties like electronegativity, radius, etc.
- These featurized compounds are visualized in a new feature space, and classification is attempted using PLS-DA.
- Most importantly, they can explore **which features are most important** for classification.

---

### Manual Feature Selection: An Interactive Learning Tool

A key part of both PCA and PLS-DA activities is the inclusion of **interactive, manual feature selection**:

- In **PCA**, students can toggle features or feature groups (e.g., radii, electronic structure) and immediately observe changes in clustering and separation on the PCA plot. This helps them build intuition about which properties matter most and how combinations of features affect element relationships.
  
- In **PLS-DA**, manual selection is similarly supported, allowing students to toggle features and evaluate classification behavior visually or through accuracy scores. They can first hypothesize, for example, “radius difference is important,” test this idea manually, and then compare their intuition to automated methods like forward or backward feature selection.

This **interactive process reinforces active learning**. Students don’t just see a pre-trained model — they **build their own**, adjust parameters, and discover chemical patterns themselves. It mimics how scientists work: forming hypotheses, testing them, and analyzing outcomes.

![plsda](https://github.com/user-attachments/assets/056ef466-79c0-48bc-a366-a77e83b697ca)


---

### Automated Feature Selection and Model Evaluation

To complement manual exploration, the PLS-DA notebook includes:
- **Forward feature selection** – Start with no features and add one at a time to optimize model accuracy.
- **Backward elimination** – Start with all features and remove the least helpful ones until reaching a minimum feature set.

Together, these tools introduce students to best practices in ML:
- **Avoiding overfitting** by limiting features or components
- **Explaining models** through smaller, meaningful feature subsets

![auto-overview](https://github.com/user-attachments/assets/f01db3fe-53a0-45df-afad-2d53b883b607)


---

### Conceptual Takeaways

By the end of the activity, students should be able to:
- Understand and explain **PCA** and **PLS-DA** from a chemical and ML perspective
- Appreciate the importance of **feature selection** and **dimensionality reduction**
- Recognize how **modern data** can be used to **revisit and validate historical chemical knowledge**
- See how **explainable ML** differs from black-box models and why that matters for science

This approach merges **data science with chemical intuition**, empowering learners to analyze real-world problems with statistical tools while grounding their understanding in physical meaning.
# Getting started

### Installation Steps

1. **Install Python**
* Download and install Python from the [official website](https://www.python.org). Follow the installer instructions for your operating system.
2. **Install Conda environment manager**
* Open your terminal (or Command Prompt on Windows) and type the following command:
```bash
pip install jupyterlab
```
  
3. **Download or Clone the Project**
*  Option 1: Download ZIP  
Download the project as a ZIP file from the repository page and extract it to a folder on your computer.

* Option 2: Clone with Git  
If you have Git installed, open your terminal and run:
```bash
git clone https://github.com/dshirya/pca-plsda-activity/tree/main
```
4. **Install Required Libraries**
* Navigate to your project folder in the terminal:
```bash
cd path/to/your/project-folder
```
* Install all necessary libraries by running:
```bash
conda create -n <name>
```
```bash
conda activate <name>
```  
```bash
pip install -r requirements.txt
```

### Running the Project

1.	Run the app on your computer: 
Once everything is installed, in the terminal go to the folder with application and:
```bash
shiny run app.py
```
You will see the link, open it in your browser:
```bash
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
2.	Deploy app to the web service:  



# Principal Component Analysis 

This notebook introduces Principal Component Analysis (PCA) as an exploratory tool, focusing on two main visualizations that build student understanding of how elemental properties can be used to infer compound behavior and classification.

### 1. PCA Element Mapping (Elements Only)

In the first part of the notebook, users load a dataset of **80 chemical elements**, each described by up to **74 numeric features**. These features are based on a wide range of physical and chemical properties (see below for the full list of groups). A **PCA analysis is performed on the element-level data**, projecting the 74-dimensional feature space into a 2D plane for visualization. The result is a **PCA-based periodic table**: each point on the scatter plot represents one element, positioned according to its projection on the first two principal components.

This part of the activity allows users to:
- **Toggle individual features or entire groups of features** (via interactive widgets)
- **Explore how feature selection impacts the PCA layout**
- **Visually identify trends or clustering among elements** based on their shared properties

This PCA periodic table helps students intuitively understand **relationships between elements** based on quantitative properties rather than periodic table position alone.

### 2. Compound clustering (Binary Compounds)

In the second part of the notebook, the same PCA space is reused to **plot binary equiatomic compounds** (from Pauling’s 1929 dataset). Here’s how it works:
- **Each compound is represented by a line** that connects the PCA coordinates of its two constituent elements (A and B).
- The **midpoint of the line is marked** to indicate the approximate PCA location of the AB compound.
- **Lines are color-coded** according to the compound’s **crystal structure type**:
  - Red: CsCl-type
  - Green: NaCl-type
  - Blue: ZnS-type

This visualization allows users to:
- **Observe how compounds with the same structure type tend to group together**
- **Test different feature subsets** to explore which combinations best separate the structure types in PCA space
- **Build intuition** about which elemental properties (e.g., radii, electronegativity) are most predictive of compound structure

### Feature Groups (74 Features Total)

The following feature categories are available for selection:
- **Basic Atomic Properties:** Atomic number, atomic weight, period, group, chemical family
- **Electronic Structure:** Mendeleev number, valence s/p/d/f electrons, unfilled orbitals, Zeff, Bohr radius, etc.
- **Electronegativity & Electron Affinity:** Multiple electronegativity scales, first ionization energy, electron affinity
- **Atomic & Ionic Radii:** Covalent, ionic, van der Waals, Slater, Miracle, polarizability, etc.
- **Thermal & Physical Properties:** Melting/boiling point, density, heat capacity, thermal conductivity, etc.
- **DFT (LDA/LSD) Properties:** E_tot, E_kin, E_coul, E_enuc, E_xc (from LDA and LSD)
- **DFT (RLDA/ScRLDA) Properties:** Analogous energy components from RLDA and ScRLDA

All features are enabled by default, but students can **individually enable/disable them** or toggle groups as a whole using buttons. This makes it easy to test hypotheses like:
- What if I only use atomic radii?
- Does removing DFT-derived features still allow for structure-type clustering?

Together, these two visualizations offer a rich, hands-on way for students to explore feature selection, dimensionality reduction, and chemical trends.

# Partial Least Square Discriminant Analysis 

This notebook introduces **Partial Least Squares–Discriminant Analysis (PLS-DA)**, a supervised machine learning method used to classify compounds into one of three structure types: **CsCl-type, NaCl-type, or ZnS-type**. In this exercise, all compounds already have class labels assigned, so the objective is to use data-driven features to train and interpret a classification model.

The features used in this notebook are generated using the **Composition Analyzer/Featurizer (CAF)** [CAF GitHub](https://github.com/bobleesj/composition-analyzer-featurizer). This tool calculates **133 compositional descriptors** derived from elemental properties of the constituent atoms in each compound. These descriptors include operations such as **sums, differences, ratios, maximum, and minimum** values of physical and chemical properties like electronegativity, atomic radius, melting point, valence electron counts, and more.

Like in the PCA notebook, students can **toggle features on and off**, exploring how different subsets of features affect structure clustering and classification accuracy. The ability to interactively control the features promotes intuition about the role each property plays in structure determination.

---

### Feature Selection (Manual and Automated)

Because 133 features may be excessive for explainability and accuracy of PLS-DA, **feature selection** is an essential step in building explainable and generalizable models.

#### Manual Feature Selection (Same Style as PCA)

Just like in the PCA notebook, this PLS-DA activity provides students with an interactive way to perform **manual feature selection** using widgets. The interface allows users to select or deselect features one by one or by **predefined feature groups**, giving full control over the input space.

Each group can be toggled on or off entirely, or users can go deeper and select individual features for inclusion in the PLS-DA model. This hands-on selection helps users explore the impact of specific types of features on classification performance and understand the **explainability** of the model.

For example, users may explore:
- Do radius difference and electronegativity difference alone suffice to separate the structures?
- Does including all ratio features improve the model or introduce noise?

This manual feature selection step promotes **active learning** and reinforces key machine learning concepts like **dimensionality**, **model simplicity**, and **feature relevance** in a chemical context.

#### Automated Feature Selection
This notebook provides both:
- **Forward Selection**
- **Backward Elimination**.

These are wrapper methods that iteratively build or reduce the feature set, measuring model performance at each step.

#### Forward Selection

Starts with two features, selected by user or randomly, then **adds features one-by-one**, each time choosing the one that most improves model performance. This continues until reaching the maximum number of features or performance plateaus.

#### Backward Elimination

Starts with all features, and **removes one-by-one**, continuing until only a minimal core remains.

These two selection methods allow students to compare how **different strategies can lead to different feature sets**, giving insight into the **robustness and redundancy** of chemical descriptors.

---

### Summary of Student Actions in This Notebook

- Load a dataset with 133 CAF-generated features and class labels
- Toggle features manually to explore clustering
- Run forward or backward feature selection to identify key descriptors
- Visualize score plots, accuracy trends, and model predictions

This interactive activity complements the PCA exercise and introduces **supervised dimensionality reduction**, **model validation**, and **explainable ML feature engineering** in an accessible way.



# References

1. Pauling, L. (1929). _The principles determining the structure of complex ionic crystals_. **J. Am. Chem. Soc.** **51**, 1010. (Linus Pauling’s original publication of his rules for crystal structures) .

2. Jaffal, E. I., Lee, S., **Shiryaev, D.**, _et al._ (2025). _Composition and structure analyzer/featurizer for explainable machine-learning models to predict solid state structures_. **Digital Discovery, 4**, 548–560. (Introduces the generation of 133 compositional features (CAF) and their use in clustering structure types with ML) .

3. BMC Bioinformatics – _So You Think You Can PLS-DA?_ (2019). [Link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3310-7). (Discussion of PLS-DA vs PCA, emphasizing PLS-DA as supervised PCA and cautions on overfitting) .

4. Hautier, G., _et al._ (2020). _The limited predictive power of the Pauling rules_. **Angew. Chem. Int. Ed.** **59**, 7569–7573. (A modern analysis of Pauling’s rules; highlights that only ~13% of structures satisfy all rules, underscoring the need for more advanced predictive methods) .

