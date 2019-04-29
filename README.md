## NCG PROJECT 
Publication Link : *[Nonlinear Conjugate Gradients For Scaling Synchronous Distributed DNN Training](https://arxiv.org/abs/1812.02886)*

This ncg repository consist of two components 
1) the **optimization** package
2) the **examples** scripts

For usafe details on the above two refer to ncg/README.md

### Getting Started
For sanity, work from with in a python virtual environment of your choice.

* Building the *optimization package* wheel
```
    # work from with in ncg directory
    python setup.py bdist_wheel -d < dist_wheel target directory>
```

* Installing the wheel in your virtual environment
```
    pip install < dist_wheel target directory>/ncg-xyz.whl
```

* Alternatively, to install ncg package in develop mode
```
    pip install -e optimization
    
```




