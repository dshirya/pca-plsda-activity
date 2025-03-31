
## Activity Overview

This repository provides an educational activity for chemistry students and professionals to learn basic machine learning techniques – specifically PCA and PLS-DA – in the context of inorganic chemistry. The activity uses **interactive Jupyter notebooks** to explore how elemental properties can be used as features for ML models. The pedagogical goal is to put a modern, data-driven spin on Linus Pauling’s 1929 study of binary compounds. In 1929, Pauling famously attempted to rationalize crystal structures of equiatomic binary compounds using simple rules based on atomic properties . This project revisits that idea with modern feature engineering and machine learning: we use a rich set of element-derived features to see if we can cluster and classify binary **AB** compounds (where A and B are elements) into their correct structure types (CsCl, NaCl, or ZnS). By combining Pauling’s early intuition with today’s ML tools, users will learn about **feature selection**, **unsupervised clustering vs. supervised classification**, and how to interpret ML results in chemical terms.

## Getting started

### Requirements

Before running the project, make sure you have the following:
* Python 3.9 or later (If you don’t have it, download it from [python.org](https://www.python.org))
* Jupyter Lab (An interactive development environment for notebooks)
* Git (Optional, if you want to clone the repository, otherwise you can download the ZIP file)
* A modern web browser (Safari, Chrome, etc.)

### Installation Steps

1. **Install Python**
* Download and install Python from the [official website](https://www.python.org). Follow the installer instructions for your operating system.
2. **Install Jupyter Lab**
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
pip install -r requirements.txt
```

### Running the Project

Once everything is installed, follow these steps to run the project:
1.	Start Jupyter Lab: 
In the terminal, make sure you are in the project folder and run:
```bash
jupyter lab
```
A web browser window will open showing the Jupyter Lab interface.

2.	Open the Notebook:  
In Jupyter Lab, locate and click on the notebook file (it will have the .ipynb extension).  
Main files:
    * ```pca-analysis.ipynb```
    * ```pls_da-analysis.ipynb```

4.	Run the Code Cells:  
You can run each cell by clicking on it and pressing **Shift + Enter** or by clicking the “Run” button. Follow the instructions in the notebook to perform PCA and PLS-DA analyses.

## PCA Analysis Notebook (`pca-analysis.ipynb`)

This notebook introduces Principal Component Analysis (PCA) as an exploratory tool, focusing on two main visualizations that build student understanding of how elemental properties can be used to infer compound behavior and classification.

### 1. PCA Periodic Table (Elements Only)

In the first part of the notebook, users load a dataset of **80 chemical elements**, each described by up to **74 numeric features**. These features are based on a wide range of physical and chemical properties (see below for the full list of groups). A **PCA analysis is performed on the element-level data**, projecting the 74-dimensional feature space into a 2D plane for visualization. The result is a **PCA-based periodic table**: each point on the scatter plot represents one element, positioned according to its projection on the first two principal components.

This part of the activity allows users to:
- **Toggle individual features or entire groups of features** (via interactive widgets)
- **Explore how feature selection impacts the PCA layout**
- **Visually identify trends or clustering among elements** based on their shared properties

This PCA periodic table helps students intuitively understand **relationships between elements** based on quantitative properties rather than periodic table position alone.

### 2. Compound Visualization (Binary Compounds)

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
