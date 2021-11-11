# Object detection evaluation

This script evaluates accuracy segmentation masks (binary|labeled) based on their corresponding groundtruths using an Intersection Over Union (IOU) or Jaccard Index based criteria.

This code calculates true positives, misses, false positives, oversegments, and undersegments in a segmentation mask. 

The input masks should be either binary masks or uint8|uint16 labelled masks. You can see the main method in the script to see how to run the evaluation code.

The general steps of the algorithm is: 

<ul>
  <li> For every object, C_i, check if at least half of its area overlaps with any ground truth object, G_j. If yes, match C_i and G_j. </li>
  <li> For every ground truth object, G_j, check if at least half of its area overlaps with any computed object, C_i. If yes, match G_j and C_i. </li>
  <li> Based on the matches, define true positive, oversegmentation, undersegmentation, false positive, and miss. </li>
  <li> Calculate precision, recall, and f1-score metrics. </li>
</ul>

   ***True positive:*** If a C<sub>i</sub> matches with exactly one ground truth object.
   
   ***Oversegmentation:*** If a C<sub>i</sub> matches with more than one groundtruth objects.
   
   ***Undersegmentation:*** If a G<sub>j</sub> matches with more than one computed objects.
   
   ***False positive:*** If C<sub>i</sub> matches with no G<sub>j</sub>.
   
   ***Miss:*** If G<sub>j</sub> matches with no C<sub>i</sub>.

Example run:
```python
if __name__ == "__main__":
    # ASSUMPTION: Computed and groundtruth masks are labeled from 1 to N (N: number of connected components)
    computed = imread("./example_data/computed.png")
    gold = imread("./example_data/gold.png")

    computed = preprocessMask(computed)
    gold = preprocessMask(gold)

    tp, overseg, underseg, miss, fp = eval(computed, gold)
    precision, recall, f1score = calculateMetrics(tp, overseg, underseg, miss, fp)

    print(f"TP:{tp}, Oversegmentation:{overseg}, Undersegmentation:{underseg}, Miss:{miss}, False positive:{fp}")
    print(f"Precision:{precision}, Recall:{recall}, F1-score:{f1score}")
```


Please cite one of the following papers if you use this code:

  [1] C. Koyuncu, G.N. Gunesli, et al., “DeepDistance: A Multi‐task Deep Regression Model for Cell Detection in Inverted
Microscopy Images”, Medical Image Analysis, 101720, 2020.
  
  [2] C. Koyuncu, Rengul Cetin‐Atalay, et al., “Object oriented cell segmentation of cell nuclei in fluorescence microscopy
images”, Cytometry Part A, 2018.
  
  [3] C. Koyuncu, E. Akhan, et al., “Iterative h‐minima based marker controlled watershed for cell nucleus segmentation”,
Cytometry Part A, 2016.
  
  [4] C. Koyuncu, S. Arslan, et al., “Smart markers for watershed‐based cell segmentation”, PloS one, 7 (11), e48664, 2012.


For any questions please contact Can Koyuncu at cfk29@case.edu
