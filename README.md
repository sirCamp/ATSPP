PROJECT FOR ADVANCED TOPICS IN SCIENTIFIC AND PARALLEL PROGRAMMING
===

*COURSE FOR PHD IN INFORMATION ENGINEERING @ UNIPD 20/21*

### BUILD THE CONTAINER
```bash
sudo singularity build container.sif container.def
```

### RUN ON GPU
```bash
sudo singularity exec --nv container.sif python /opt/workspace/script.py
```

### RUN ON CPU
```bash
sudo singularity exec container.sif python /opt/workspace/script.py
```