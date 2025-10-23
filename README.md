# CG2 Homework 1 — Computer Graphics II (Fall 2025)

**Author:** Chengyan Guo  
**Student ID:** 2025233151  
**Institution:** ShanghaiTech University  
**Course:** CS271 Computer Graphics II  
**Submission Date:** October 23, 2025  

## Overview

This repository contains the complete implementation and report materials for **Homework 1** of *Computer Graphics II*.  
The assignment consists of two independent computational geometry problems:

1. **Problem 1 — Melkman’s Algorithm for Simple Polygon Convex Hull**  
   Implementation and analysis of an $O(n)$ convex hull algorithm for simple polygons.

2. **Problem 2 — General Voronoi (Power Diagram)**  
   Implementation and visualization of a weighted Voronoi diagram (Power Diagram) based on half-plane intersections, including theoretical and empirical complexity analysis.


## Directory Structure
```
CG2_HW1/
├── chengyan_guo_2025233151_solution.pdf     # Final compiled report for submission
├── README.md                                # Project documentation (this file)
│
├── document/                                # Report and related resources
│   ├── report_template.tex                  # LaTeX report source file
│   ├── HW1.pdf                              # Homework description (reference)
│   ├── General Voronoi.pdf                  # Lecture slides (reference)
│   └── Pictures/                            # Figures used in the report
│       ├── hulls.png
│       ├── melkman_complexity.png
│       ├── polygon_cases.png
│       ├── power_complexity1.png
│       ├── power_complexity2.png
│       ├── power_diagram.png
│       ├── sites_cases.png
│       ├── test_hulls.png
│       ├── test_pd1.png
│       ├── test_pd2.png
│       ├── test_polygon_cases.png
│       ├── test_sites1.png
│       └── test_sites2.png
│
├── Problem1Code/                            # Problem 1: Melkman Convex Hull
│   ├── polygon_generate_suite.py            # Simple polygon generation
│   ├── poly_cases.json                      # Predefined polygon vertex sets
│   ├── viz_suite.py                         # Visualization utilities for polygons and hulls
│   ├── complexity_analysis.py               # Empirical complexity experiment 
│   └── Problem1.ipynb                       # Integrated notebook for implementation and analysis
│
└── Problem2Code/                            # Problem 2: General Voronoi / Power Diagram
    ├── voronoi_site_suite.py                # Weighted-site generation
    ├── voronoi_cases.json                   # Predefined site configurations
    ├── viz_suite.py                         # Visualization utilities for Power Diagram rendering
    ├── complexity_power.py                  # Empirical complexity experiment
    └── Problem2.ipynb                       # Integrated notebook for implementation and analysis
```

## Usage

### 1. Run the Notebooks
Open and execute the following notebooks in order:
```bash
Problem1Code/Problem1.ipynb
Problem2Code/Problem2.ipynb
```

### 2. Compile the LaTeX Report
```bash
cd document
latexmk -pdf report_template.tex
```

The compiled report will appear as report_template.pdf (or the precompiled chengyan_guo_2025233151_solution.pdf).

## Environment

| Component | Version / Description |
|------------|-----------------------|
| Python | 3.9 |
| Libraries | `numpy`, `matplotlib`, `math`, `json`, `collections`, `pathlib` |
| Notebook | JupyterLab / VS Code Jupyter |
| LaTeX | `latexmk` or `pdflatex` |
| OS | macOS / Linux (tested) |


## Results Overview

| Problem | Algorithm | Theoretical Time | Empirical Trend | Visualization |
|----------|------------|------------------|-----------------|----------------|
| 1 | Melkman Convex Hull | $O(n)$ | Linear scaling | `polygon_cases.png`, `melkman_complexity.png` |
| 2 | Power Diagram | $O(n^2)$ | Quadratic scaling | `power_diagram.png`, `power_complexity*.png` |