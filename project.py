{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPbrjjMIzzP0J2MQ3kKUtCf",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sameer-6025/Energy-Efficiency-Optimazation/blob/main/project.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-lRCS4PTtDND"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "# Simulate IoT energy data\n",
        "np.random.seed(42)\n",
        "days = np.arange(1, 101)  # 100 days\n",
        "energy_usage = 100 + 10 * np.sin(0.2 * days) + np.random.normal(0, 5, size=100)\n",
        "\n",
        "# Create DataFrame\n",
        "df = pd.DataFrame({\n",
        "    'Day': days,\n",
        "    'EnergyUsage_kWh': energy_usage\n",
        "})\n",
        "\n",
        "# Plot raw data\n",
        "plt.figure(figsize=(10, 5))\n",
        "plt.plot(df['Day'], df['EnergyUsage_kWh'], label='Actual Energy Usage', color='blue')\n",
        "plt.title('IoT Energy Usage Data (100 Days)')\n",
        "plt.xlabel('Day')\n",
        "plt.ylabel('Energy Usage (kWh)')\n",
        "plt.grid(True)\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n",
        "# Prepare data for prediction\n",
        "X = df[['Day']]\n",
        "y = df['EnergyUsage_kWh']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)\n",
        "\n",
        "# Train ML model\n",
        "model = LinearRegression()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict and evaluate\n",
        "y_pred = model.predict(X_test)\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "print(f\"Mean Squared Error: {mse:.2f}\")\n",
        "\n",
        "# Predict future energy usage (next 10 days)\n",
        "future_days = np.arange(101, 111).reshape(-1, 1)\n",
        "future_predictions = model.predict(future_days)\n",
        "\n",
        "# Plot predictions\n",
        "plt.figure(figsize=(10, 5))\n",
        "plt.plot(df['Day'], df['EnergyUsage_kWh'], label='Historical Data', color='blue')\n",
        "plt.plot(future_days, future_predictions, label='Predicted Usage', color='red', linestyle='--')\n",
        "plt.title('Energy Usage Forecast')\n",
        "plt.xlabel('Day')\n",
        "plt.ylabel('Energy Usage (kWh)')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ]
    }
  ]
}