import java.util.*;
import java.io.*;
import org.ejml.simple.*;


public class Datum {
	private SimpleMatrix allX;
	private SimpleMatrix allY;
	
	private int numRowX;
	private int numColX;
	private int numRowY;
	private int numColY;
	
	public Datum(int numRowX, int numColX, int numRowY, int numColY) {		
		this.numRowX = numRowX;
		this.numColX = numColX;
		this.numRowY = numRowY;
		this.numColY = numColY;
		
		allX = new SimpleMatrix(this.numRowX, this.numColX);
		allY = new SimpleMatrix(this.numRowY, this.numColY);
	}
	
	
	public void readData(String filePath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filePath));
		ArrayList<String> lineList = new ArrayList<String>();
		try {
			String line = br.readLine();
			while(line != null) {
				lineList.add(line);
				line = br.readLine();
			}
		} finally {
			br.close();
		}
		
		for(int i=0; i < lineList.size(); i++) {
			String[] lineArray = lineList.get(i).split(",");
			//System.out.println(lineList.get(i));
			for(int j=0; j < this.numRowX; j++) {
				allX.set(j, i, Double.parseDouble(lineArray[j]));
			}
			if (numRowY == 1)
				allY.set(0, i, Double.parseDouble(lineArray[this.numRowX]));
			else {
				for (int j=0; j < numRowY; j++) {
					if (j == Integer.parseInt(lineArray[this.numRowX]) - 1)
						allY.set(j, i, 1.0);
					else
						allY.set(j, i, 0.0);
				}
			}
//			if (lineArray[this.numRowX].equals("1")) {
//				allY.set(0, i, 0.0);
//				allY.set(1, i, 1.0);
//			}
//			else {
//				allY.set(0, i, 1.0);
//				allY.set(1, i, 0.0);
//			}
		}
	}
	
	public SimpleMatrix getAllX() {
		return this.allX;
	}
	public SimpleMatrix getAllY() {
		return this.allY;
	}
}
