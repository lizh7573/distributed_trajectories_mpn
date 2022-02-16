from setuptools import find_packages, setup

setup(
    author='Liyuan Zhang',
    author_email='liyuan.zhang.7573@student.uu.se',
    name='distributed_trajectories_mpn',
    version='0.1.0',
    install_requires=[],
    license='Apache License',
    packages=find_packages(include=['distributed_trajectories_mpn', 'distributed_trajectories_mpn.*']),
    include_package_data=True,
    setup_requires=['pytest_runner'],
    tests_require=['pytest'],
    test_suite='tests',
    url='https://github.com/lizh7573/distributed_trajectories_mpn',
)