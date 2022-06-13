import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCRT",
    version="2.0.2",
    author="Eduardo Lopes Dias",
    author_email="eduardosprp@protonmail.com",
    description="Tools for measuring Capillary Refill Time using Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Photobiomedical-Instrumentation-Group/pyCRT",
    project_urls={
        "Bug Tracker": "https://github.com/Photobiomedical-Instrumentation-Group/pyCRT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python",
        "matplotlib",
        "scipy",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
