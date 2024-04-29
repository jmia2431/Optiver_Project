from palmerpenguins import load_penguins
from shiny import App, render, ui

# Load your data (replace 'penguins' with your actual data)
penguins = load_penguins()

# Define the UI for the app
app_ui = ui.page_fluid(
    ui.h2("Palmer Penguins"),
    ui.output_data_frame("penguins_df"),
)

# Define the server logic
def server(input, output, session):
    @render.data_frame
    def penguins_df():
        return render.DataTable(penguins, row_selection_mode="single")

# Create the App Instance
app = App(app_ui, server)

# Run the App
if __name__ == "__main__":
    app.run()
