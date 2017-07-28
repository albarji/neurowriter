FROM continuumio/anaconda3

# Install requirements
ADD conda.txt pip.txt ./
RUN conda install -y --file conda.txt && rm conda.txt
RUN pip install -r pip.txt && rm pip.txt

# Clone project files
RUN mkdir neurowriter
COPY . /neurowriter
WORKDIR /neurowriter

# Launche Jupyter notebook with appropriate options
CMD jupyter notebook --allow-root --no-browser

