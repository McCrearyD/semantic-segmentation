# Comma Hackathon Semantic-Segmentation

### How to setup:
- Build the docker container: `sudo docker build -t segsem .`
- Run the docker container: `sudo docker run --shm-size {ALLOCATE_MEMORY_BITES} -it -p 8080:8080 --gpus all segsem`
    - Optionally, you can mount a volume to the docker container for live code changes: `sudo docker run --shm-size {ALLOCATE_MEMORY_BITES} -v /path/to/semantic-segmentation:/home/semantic-segmentation -it -p 8080:8080 --gpus all segsem`