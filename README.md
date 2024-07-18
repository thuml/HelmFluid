# HelmFluid: Learning Helmholtz Dynamics for Interpretable Fluid Prediction (ICML 2024)

HelmFluid: Learning Helmholtz Dynamics for Interpretable Fluid Prediction [[paper]](https://arxiv.org/pdf/2310.10565)

Inspired by the Helmholtz theorem, we propose HelmFluid to learn curl-free and divergence-free parts of the fluid field, 

- Inspired by the Helmholtz theorem, we propose the *Helmholtz dynamics* to attribute intricate dynamics into inherent properties of fluid, which **empowers the prediction process with physical interpretability**.
-  We propose HelmFluid with the *HelmDynamics block* to capture Helmholtz dynamics. By integrating learned dynamics along temporal dimension with the *Multiscale Multihead Integral Architecture*, HelmFluid can predict future fluid with **physically plausible evidence**.
- HelmFluid achieves consistent state-of-the-art in five extensive benchmarks, covering **both synthetic and realworld datasets**, as well as **various boundary conditions**. 

<p align="center">
<img src=".\fig\model.jpg" height = "220" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of HelmFluid.
</p>

## HelmFluid vs. Previous Methods

- Different from previous methods that directly learning fluid dynamics, HelmFluid decomposes the intricate dynamics into more solvable parts, which facilitates our model with physical interpretability.
- Different from neural fluid simulators like PDE solvers, HelmFluid is as purely data-driven model but with special designs to enhance physical interpretability.
- Unlike computer graphics for fluid simulation, HelmFluid is an end-to-end method to learn intrincate dynamics without ground truth velocity supervision nor stream function.


<p align="center">
<img src=".\fig\compare.jpg" height = "150" alt="" align=center />
<br><br>
<b>Figure 2.</b> Comparison on dynamics and fluid modeling.
</p>

## Get Started

1. Install Python 3.8. For convenience, execute the following command. You may change the version of cupy according to your own environment.

```bash
pip install -r requirements.txt
```

2. Data preparation.
Download the datasets from the following links and put them under the folder `./data/`.

| Dataset         | Task                           |  Link                                                        |
|-----------------|--------------------------------| ----------------------------------------------------------- |
| Navier-Stokes   | Predict future fluid vorticity | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Bounded N-S     | Predict future color           | [[Google Cloud]](https://drive.google.com/drive/folders/127ki3Oo8xt1KVfjgtsRqeuf853jgf98E) |
| ERA5 Z500       | Predict future geopotential    | [[Google Cloud]](https://drive.google.com/drive/folders/127ki3Oo8xt1KVfjgtsRqeuf853jgf98E) |
| Sea Temperature | Predict future sea temperature | [[Google Cloud]](https://drive.google.com/drive/folders/127ki3Oo8xt1KVfjgtsRqeuf853jgf98E) |
| Spreading Ink   | Predict future fluid video     | [[Google Cloud]](https://drive.google.com/file/d/1kCX2NF4IMtB2IC_xZPfrrnR2uiV83zRC/view) |

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:
```bash
bash scripts/Bounded_NS_HelmFluid.sh # for Bounded N-S
bash scripts/ERA5_Z500_HelmFluid.sh # for ERA5_Z500
bash scripts/Navier_Stokes_HelmFluid.sh # for Navier-Stokes
bash scripts/Sea_Temperature_HelmFluid.sh # for Sea Temperature
bash scripts/SpreadingInk_HelmFluid.sh # for Spreading Ink
```

 Note: You need to change the argument `--data-path` in the above script files to your dataset path.

4. Develop your own model. Here are the instructions:
   - Add the model file under folder `./models/`.
   - Add the model name into `./model_dict.py`.
   - Add a script file under folder `./scripts/` and change the argument `--model`.

 Note: For clearness and easy comparison, we also include the FNO in this repository.

## Results

We extensively experiment on seven benchmarks and compare LSM with 13 baselines. LSM achieves the consistent state-of-the-art in both solid and fluid physics (11.5% averaged error reduction).

<p align="center">
<img src=".\fig\NSresults.jpg" height = "200" alt="" align=center />
<br><br>
<b>Table 1.</b> Model performance for Navier-Stokes dataset. Relative L2 is recorded.
</p>

## Showcases

<p align="center">
<img src=".\fig\showcasens.jpg" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases for Navier-Stokes dataset. HelmFluid precisely predicts the fluid motion, especially the twist parts.
</p>


<p align="center">
<img src=".\fig\showcaseBoundedns.jpg" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases for Bounded N-S dataset. HelmFluid succesfully captures the Karmen vortex phenomenon.
</p>


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{xing2024HelmFluid,
  title={HelmFluid: Learning Helmholtz Dynamics for Interpretable Fluid Prediction},
  author={Lanxiang Xing and Haixu Wu and Yuezhou Ma and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [xlx22@mails.tsinghua.edu.cn](mailto:xlx22@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/yitongdeng-projects/learning_vortex_dynamics_code

https://github.com/erizmr/Learn-to-Estimate-Fluid-Motion
