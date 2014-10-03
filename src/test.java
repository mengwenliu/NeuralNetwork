import org.ejml.simple.*;

public class test {
	public static void main(String[] args) {
		SimpleMatrix a = new SimpleMatrix(2,3);
		a.set(2.0);
		System.out.println("sum of square of elements in a: " + a.elementMult(a).elementSum());
		System.out.println("a.elementSum(): " + a.elementSum());
		
		SimpleMatrix b = a.copy();
		b.set(1.0);
		System.out.println("b[0][0]: " + b.get(0, 0));
		System.out.println("a[0][0]: " + a.get(0, 0));
		
		b = b.extractMatrix(1, b.numRows(), 0, b.numCols());
		//b = bNew.copy();
		System.out.println("b.size() after extracting the first row: " + b.numRows() + "*" + b.numCols());

	}

}
