import setuptools

setuptools.setup(
    name="deep_rl_box",
    version="0.0.1",
    author="Sam Zamani",
    author_email="sam.zmn99@gmail.com",
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[],
    description="a deep reinforcement learning framework",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)