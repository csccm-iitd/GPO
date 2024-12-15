from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename,"r") as f:
        return f.read().splitlines()
    

setup(
    name = "GPO-torch",
    version = "0.1",
    description = "Gaussian Process Operator in PyTorch and GPyTorch",
    author = "sk",
    packages = find_packages(),
    install_requires = parse_requirements("reuirements.txt"),
    classifiers=[
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',

)