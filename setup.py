from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Development of a demand forecasting model for an American public transport company to optimize vehicle allocation. Using six months of GPS and vehicle start-time data, the goal is to predict hourly taxi demand in the ten most important city clusters. The project involves data quality assessment, clustering, regression modeling per cluster, error analysis, and proposing an integration strategy for daily logistic operations.',
    author='sandra_gedig',
    license='MIT',
)
