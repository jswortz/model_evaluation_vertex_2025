# Fetch the base image
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

# Set the working dir for the rest of the commands
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
