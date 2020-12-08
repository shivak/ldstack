## Linear Dynamical Systems as a Core Computational Primitive

This is a reference implementation of the <a href="https://proceedings.neurips.cc//paper_files/paper/2020/hash/c3581d2150ff68f3b33b22634b8adaea-Abstract.html">eponymous paper</a> at NeurIPS 2020.
The main construction is **LDStack**, a drop-in replacement for nonlinear RNNs (e.g. LSTMs) which (1) runs in parallel across the sequence length, and (2) is easier to interpret and analyze, since it is composed from linear systems. 
The core computational primitive: single-input, multiple-output **(SIMO) LDS** parameterized in terms of complex eigenvalues. These have a parallel CUDA implementation. 
The <a href="./NeurIPS2020_experiments.ipynb">Jupyter notebook</a> reproduces the experiments provided in the paper.

The broad goal: make RNNs **fast** (on long sequences), **accurate** (with nonlinear expressive power) *and* **trustworthy** (via simpler mathematical and intuitive analysis). 
If you are a researcher or practitioner interested in trustworthy machine learning models, <a href="mailto:skkaul@cs.cmu.edu">reach out to me</a> and I will help you apply this method.

### Frequently Asked Questions

*How do I build the CUDA op?* Use the `build.sh` script in the `linear_recurrent_net` folder. We will soon provide a PyPI package that you can `pip install`.

*Is there a PyTorch version?* Soon! 

