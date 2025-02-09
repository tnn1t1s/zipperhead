from setuptools import setup, find_packages

setup(
    name="zipperhead",
    version="0.1",
    packages=find_packages(where="src"),  # Finds packages in src/
    package_dir={"": "src"},  # Tells setuptools that packages live in src/
    install_requires=[
        "numpy",
        "torch",
        "pytest",
        "jupyter",
        "scipy"
    ],
)

