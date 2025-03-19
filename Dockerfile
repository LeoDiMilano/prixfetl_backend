FROM python:3.11

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    cron \
    sudo \
    python3-pip \
    openssh-server \
    wget \
    unzip \
    libx11-xcb1 libxcomposite1 libxi6 libxrandr2 libasound2 \
    libpangocairo-1.0-0 libatk1.0-0 libcups2 libnss3 libxss1 \
    libgconf-2-4 libxdamage1 libxshmfence1 libglu1-mesa \
    libxtst6 fonts-liberation libappindicator3-1 libdbusmenu-glib4 \
    libdbusmenu-gtk3-4 libgtk-3-0 \
    libgbm1 libvulkan1 xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Configurer OpenSSH
RUN mkdir -p /var/run/sshd
RUN useradd -m -s /bin/bash ubuntu && echo "ubuntu:Leumcesnl120674" | chpasswd
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && chmod 0440 /etc/sudoers.d/ubuntu
RUN echo "root:YhtDn58dmSimple\$dsjfTjdnkbv5" | chpasswd

# Télécharger et installer Google Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb

# Télécharger et installer Chromedriver
RUN CHROMEDRIVER_VERSION=$(wget -qO- https://chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip && \
    mv chromedriver /usr/local/bin/ && \
    chmod +x /usr/local/bin/chromedriver && \
    rm chromedriver_linux64.zip

# Copier la configuration cron
COPY crontab /etc/cron.d/ia-cron
RUN chmod 0644 /etc/cron.d/ia-cron
RUN crontab /etc/cron.d/ia-cron

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Installer Python packages
COPY . /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:/app/routers:/app/services"
# Installer Jupyter Notebook
#RUN pip install --no-cache-dir jupyter

# Configurer Git dans le conteneur
RUN git config --global user.name "LeoDiMilano" && \
    git config --global user.email "leonardgallone@outlook.fr"

# Configurer le service SSH et exécuter Flask
CMD ["/entrypoint.sh"]

