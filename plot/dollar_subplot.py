from plot.base_model import SubplotBase


class DollarSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)

        self.current_dollars = []
        self.dollars_line, = self.ax.plot([], [], color='brown', label='Current Dollars')
        self.lines.append(self.dollars_line)
        self.ax.legend()

    def update_limits(self):
        if len(self.steps) > 1:  # Asegúrate de que hay al menos dos pasos para establecer límites
            self.ax.set_xlim(self.steps[0], self.steps[-1])
        # Set the limits of the x-axis and y-axis
        if len(self.steps) > 30:
            self.ax.set_xlim(self.steps[-30], self.steps[-1])
            
            # Set the y-axis limits to show a bit more space above and below the max and min
            y_values = self.current_dollars[-30:]
            if y_values:
                min_val, max_val = min(y_values), max(y_values)
                if min_val == max_val:  # Avoid having a flat line by expanding the limits slightly
                    min_val -= 0.05 * abs(max_val)  # Adjust min_val to 5% below max_val if they are equal
                    max_val += 0.05 * abs(max_val)  # Adjust max_val to 5% above itself if they are equal
                self.ax.set_ylim(min_val, max_val)
        else:
            self.ax.set_xlim(self.steps[0], self.steps[-1])
            y_values = self.current_dollars
            if y_values:
                min_val, max_val = min(y_values), max(y_values)
                if min_val == max_val:
                    min_val -= 0.05 * abs(max_val)
                    max_val += 0.05 * abs(max_val)
                self.ax.set_ylim(min_val, max_val)
        
        self.ax.relim()  # Recalculate the limits based on the current data
        self.ax.autoscale_view()  # Rescale the view to the new limits

    def plot(self, current_dollars, step):
        self.add_step(step)
        self.current_dollars.append(current_dollars)
        self.dollars_line.set_data(self.steps, self.current_dollars)
        self.update_limits()

    def reset(self):
        super().reset()
        self.current_dollars = []