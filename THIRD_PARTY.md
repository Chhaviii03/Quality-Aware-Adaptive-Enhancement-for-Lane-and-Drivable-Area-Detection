# Third-Party Notices and Acknowledgements

This project uses subsets of datasets and resources provided by third parties.

---

## 1. BDD100K Dataset

- **Title**: BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
- **Authors**: Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, Trevor Darrell
- **Institution**: UC Berkeley / Berkeley Artificial Intelligence Research (BAIR)
- **Publication**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020
- **URL**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **License**: BDD100K Database License Agreement (Non-Commercial Research and Educational Use Only)

### Citation
```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2637--2646},
  year={2020}
}
```

---

## 2. Hugging Face Mirror

- **Mirror Repository**: [`dronefreak/BDD100K-Drivable-Area-Segmentation`](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation)
- **Type**: Unofficial redistribution / mirror of BDD100K drivable area segmentation data and images
- **Note**: The subset of images and drivable-area ground-truth masks sampled by this project is pulled from this mirror for local development and non-commercial research purposes in accordance with the upstream BDD100K license.

---

## 3. Reimplemented Scientific Algorithms & References

All algorithms in this project are clean reimplementations based on published academic literature:

- **Dark Channel Prior & Guided Filter (Dehazing)**:
  - *He, K., Sun, J., & Tang, X. (2011)*. Single Image Haze Removal Using Dark Channel Prior. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 33(12), 2341-2353.
  - *He, K., Sun, J., & Tang, X. (2013)*. Guided Image Filtering. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 35(6), 1397-1409.
- **Fast Noise Variance Estimation**:
  - *Immerkaer, J. (1996)*. Fast Noise Variance Estimation. *Computer Vision and Image Understanding*, 64(2), 300-302.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**:
  - *Zuiderveld, K. (1994)*. Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*, Academic Press Professional, Inc., 474-485.
- **Koschmieder Atmospheric Scattering Model (Synthetic Fog)**:
  - *Koschmieder, H. (1924)*. Theorie der horizontalen Sichtweite. *Beiträge zur Physik der freien Atmosphäre*, 12, 33-53.
- **Non-Local Means Denoising**:
  - *Buades, A., Coll, B., & Morel, J. M. (2005)*. A non-local algorithm for image denoising. *IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2, 60-65.

