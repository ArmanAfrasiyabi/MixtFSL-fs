#  [Mixture-based Feature Space Learning for Few-shot Image Classification](https://lvsn.github.io/MixtFSL/) 
 
This repository contains the pytorch implementation of Mixture-based Feature Space Learning for Few-shot Image Classification [paper](https://arxiv.org/abs/1912.05094) 
[presentation](https://lvsn.github.io/MixtFSL/assets/MixFSL_Poster.pdf). This paper introduces Mixture-based Feature Space Learning (MixtFSL) for obtaining a rich and robust feature representation in the context of few-shot image classification. Previous works have proposed to model each base class either with a single point or with a mixture model by relying on offline clustering algorithms. In contrast, we propose to model base classes with mixture models by simultaneously training the feature extractor and learning the mixture model parameters in an online manner. This results in a richer and more discriminative feature space which can be employed to classify novel examples from very few samples. 
 
## Dependencies
1. numpy
2. Pytorch 1.0.1+ 
3. torchvision 0.2.1+
4. PIL


## Train 
1. Hyper-parameters and training details are specified in <code>args_parser.py</code>. 
2. Run main from <code>main.py</code> to capture and test the best model in ./checkpoints.



## Datasets
- Download miniImageNet dataset from here [here](https://drive.google.com/file/d/1G1fcf8WdbqW0atc98KXF3UgVZyaXXJo4/view?usp=sharing), 
and copy the dataset in your perfered directory. 
- Run <code>write_base.py </code>, <code>write_val.py </code>, and <code>write_novel.py </code> under the directory. 
- Please don't forget to specify the dataset **directory** in <code>args_parser.py</code>. 




## The project webpage
Please visit [the project webpage](https://lvsn.github.io/MixtFSL/) for more information.

## Citation
</code><pre>
@InProceedings{Afrasiyabi_2021_ICCV,
    author    = {Afrasiyabi, Arman and Lalonde, Jean-Fran{\c{c}}ois and Gagn{\'e}, Christian},
    title     = {Mixture-Based Feature Space Learning for Few-Shot Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9041-9051}
} 
</code></pre>
