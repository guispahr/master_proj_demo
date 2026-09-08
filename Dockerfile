FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

USER root

# Create group and user matching your EPFL account
RUN groupadd -g ${LDAP_GID} ${LDAP_GROUPNAME} && \
    useradd -m -u ${LDAP_UID} -g ${LDAP_GID} -s /bin/bash ${LDAP_USERNAME}

RUN apt-get update && apt-get install -y sudo && \
    echo "${LDAP_USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${LDAP_USERNAME} && \
    chmod 0440 /etc/sudoers.d/${LDAP_USERNAME} && \
    rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="7.0;8.0"

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-build-isolation flash-attn spconv-cu126==2.3.8 \
    && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

COPY . /app

WORKDIR /app/libs/pointrope
RUN pip install -e .

WORKDIR /app

USER ${LDAP_USERNAME}
CMD ["python"] 

