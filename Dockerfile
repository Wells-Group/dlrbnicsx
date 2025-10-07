# Pull dolfinx 0.7.3 docker image
FROM dolfinx/dolfinx:v0.7.3

# Set work directory
RUN mkdir -p /home/dlrbnicsx
WORKDIR /home/dlrbnicsx

RUN pip3 install git+https://github.com/RBniCS/ufl4rom.git

# RUN pip3 install git+https://github.com/RBniCS/RBniCSx.git

RUN pip3 install git+https://github.com/niravshah241/rbnicsx_for_dlrbnicsx.git

RUN pip3 install git+https://github.com/niravshah241/MDFEniCSx.git

RUN pip3 install torch

RUN pip3 install matplotlib

RUN pip3 install plotly

RUN pip3 install git+https://github.com/Wells-Group/dlrbnicsx.git

# Keep container alive
CMD ["sleep", "infinity"]