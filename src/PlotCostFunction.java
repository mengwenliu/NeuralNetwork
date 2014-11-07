import javax.swing.*;

import org.math.plot.*;

public class PlotCostFunction {
	public void plotCurves(double[] x, double[] y) {
		Plot2DPanel plot2D = new Plot2DPanel();
		plot2D.addLinePlot("my plot", x, y);
		
		// put the PlotPanel in a JFrame, as JPanel
		JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
		frame.setContentPane(plot2D);
		frame.setVisible(true);
	}
	
	//testing 
	public static void main(String[] args) {
		 double[] x = new double[]{0,1,2,3,4,5,6,7,8,9,10};
		 double[] y = new double[]{0.9,0.85,0.7,0.6,0.5,0.4,0.33,0.2,0.19,0.15,0.1};
		 
		Plot2DPanel plot2D = new Plot2DPanel();
		plot2D.addLinePlot("my plot", x, y);
		
		// put the PlotPanel in a JFrame, as JPanel
		JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);

		frame.setContentPane(plot2D);
		frame.setVisible(true);
	}

}
