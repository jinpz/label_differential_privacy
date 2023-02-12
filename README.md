# Does Label Differential Privacy Prevent Label Inference Attacks?

This repo contains the official implementation for the paper [Does Label Differential Privacy Prevent Label Inference Attacks?](https://arxiv.org/abs/2202.12968). In this paper, we analyze in depth the connection between label differential privacy and label inference attacks with both theoretical and empirical results.

## Environment Setup
To install required packages, run the command below (we recommend python 3.7 and pytorch 1.7):

```setup
pip install -r requirements.txt
```

## Folder Structure
```
criteo/
    data/
    LP-MST/
    PATE/
image/
    LP-MST/
    PATE/
simulation/
  ```

`image/` contains experiment code for Section 3.

`simulation/` contains experiment code for Section 5.1.

`criteo/` contains experiment code for Section 5.2.

Please see individual folder for more information.

## Citation
If you find this code useful in your research, please consider citing:

    @inproceedings{wu2022does,
      title={Does Label Differential Privacy Prevent Label Inference Attacks?},
      author={Wu, Ruihan and Zhou, Jin Peng and Weinberger, Kilian Q and Guo, Chuan},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      year={2023}
    }
