# Import Shiny
from shiny import App, reactive, render, ui

# Define the User Interface (UI)
app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            # Button for Clustering
            ui.input_action_button("btn_clustering", "Clustering"),
            # Button for KMeans
            ui.input_action_button("btn_kmeans", "KMeans"),
            # Button for ARMA GARCH
            ui.input_action_button("btn_arma_garch", "ARMA GARCH"),
            # Button for Arima
            ui.input_action_button("btn_arima", "Arima Model"),
        ),
        ui.panel_main(
            ui.navset_tab_card(
                ui.nav("Description", ui.output_text("description")),
                ui.nav("Plot", ui.output_plot("plot")),
                ui.nav("Output", ui.output_text("output"))
            )
        )
    )
)

# Define the Server Logic
def server(input, output, session):
    # Reactive variable to track which button was pressed
    current_tab = reactive.Value("")

    @reactive.Effect
    def update_tab():
        # Change the current tab based on which button is pressed
        if input.btn_clustering():
            current_tab.set("Clustering")
        elif input.btn_kmeans():
            current_tab.set("KMeans")
        elif input.btn_arma_garch():
            current_tab.set("ARMA GARCH")
        elif input.btn_arima():
            current_tab.set("Arima Model")

    # Render the description tab based on the current button pressed
    @output
    @render.text
    def description():
        tab = current_tab.get()
        if tab == "Clustering":
            return "Description for Clustering method."
        elif tab == "KMeans":
            return "Description for KMeans method."
        elif tab == "ARMA GARCH":
            return "Description for ARMA GARCH method."
        elif tab == "Arima Model":
            return "Description for Arima Model."
        else:
            return "Select a button to get started."

    # Render a plot based on the current button pressed
    @output
    @render.plot
    def plot():
        import matplotlib.pyplot as plt
        import numpy as np
        
        tab = current_tab.get()
        if tab == "Clustering":
            # Example plot for Clustering
            x = np.random.rand(100)
            y = np.random.rand(100)
            plt.scatter(x, y, c='b', label='Clustering')
            plt.title("Clustering Plot")
        elif tab == "KMeans":
            # Example plot for KMeans
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            plt.plot(x, y, 'r-', label='KMeans')
            plt.title("KMeans Plot")
        elif tab == "ARMA GARCH":
            # Example plot for ARMA GARCH
            x = np.linspace(0, 10, 100)
            y = np.cos(x)
            plt.plot(x, y, 'g-', label='ARMA GARCH')
            plt.title("ARMA GARCH Plot")
        elif tab == "Arima Model":
            # Example plot for Arima Model
            x = np.linspace(0, 10, 100)
            y = np.tan(x)
            plt.plot(x, y, 'm-', label='Arima Model')
            plt.title("Arima Model Plot")
        
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.show()

    # Render the output tab based on the current button pressed
    @output
    @render.text
    def output():
        tab = current_tab.get()
        if tab == "Clustering":
            return "Output for Clustering."
        elif tab == "KMeans":
            return "Output for KMeans."
        elif tab == "ARMA GARCH":
            return "Output for ARMA GARCH."
        elif tab == "Arima Model":
            return "Output for Arima Model."
        else:
            return "Select a button to view the output."

# Create the App Instance
app = App(app_ui, server)

# Run the App
if __name__ == "__main__":
    app.run()
