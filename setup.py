from setuptools import setup, find_packages

setup(
    name="CI singles",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "quantum-systems @ git+https://github.com/Schoyen/quantum-systems",
    ],
)
