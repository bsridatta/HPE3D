import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HPE3D_bsridatta",
    version="0.0.1",
    author="Sri Datta Budaraju",
    author_email="b.sridatta@gmail.com",
    description="Unsupervised 3D human pose estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bsridatta/HPE3D",
    packages=setuptools.find_packages(),
    classifiers=[      
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
