// Source code from: https://github.com/yusugomori/DeepLearning
// original author: Yusuke Sugomori

public class LogisticLR {
	public static void main(String[] args) {
		
		// Training parameter
		double learning_rate = 0.1;
        int n_epochs = 500;
        int train_N = 6;
        int test_N = 2;
        int n_in = 6;
        int n_out = 2;
        
        
        // Array of training data (input)
        double[][] train_X = {
                {1., 1., 1., 0., 0., 0.},
                {1., 0., 1., 0., 0., 0.},
                {1., 1., 1., 0., 0., 0.},
                {0., 0., 1., 1., 1., 0.},
                {0., 0., 1., 1., 0., 0.},
                {0., 0., 1., 1., 1., 0.}
        };
        
     // Array of training data (label)
        int[][] train_Y = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };
        
        
        // Construct the LogisticRegression object
        LogisticRegression classifier = new LogisticRegression(train_N, n_in, n_out);
        
        // Training parameter
        for(int epoch=0; epoch<n_epochs; epoch++) {
            for(int i=0; i<train_N; i++) {
            	// Call the train function
                classifier.train(train_X[i], train_Y[i], learning_rate);
            }
        }

        
        // Generating test data (input)
        double[][] test_X = {
                {1., 0., 1., 0., 0., 0.},
                {0., 0., 1., 1., 1., 0.}
        };
        
        
        // Generating test data (label)
        double[][] test_Y = new double[test_N][n_out];


        
        for(int i=0; i<test_N; i++) {
            classifier.predict(test_X[i], test_Y[i]);
            for(int j=0; j<n_out; j++) {
                System.out.print(test_Y[i][j] + " ");
            }
            System.out.println();
        }             
	}

}






class LogisticRegression {
    public int N;
    public int n_in;
    public int n_out;
    public double[][] W;
    public double[] b;
    
    
    // initialization
    public LogisticRegression(int N, int n_in, int n_out) {
        this.N = N;
        this.n_in = n_in;
        this.n_out = n_out;

        W = new double[n_out][n_in];
        b = new double[n_out]; //bias
    }
    
    
    // Using Softmax function as loss
    public void softmax(double[] x) {
        double max = 0.0;
        double sum = 0.0;
        
        // Find max value
        for(int i=0; i<n_out; i++) if(max < x[i]) max = x[i];

        for(int i=0; i<n_out; i++) {
            x[i] = Math.exp(x[i] - max); // minus max for avoiding some error
            sum += x[i];
        }

        for(int i=0; i<n_out; i++) x[i] /= sum;
    }
    
    
   
    
    // Training 
    public double[] train(double[] x, int[] y, double lr) {
        double[] p_y_given_x = new double[n_out];
        double[] dy = new double[n_out];

        for(int i=0; i<n_out; i++) {
            p_y_given_x[i] = 0;
            for(int j=0; j<n_in; j++) {
                p_y_given_x[i] += W[i][j] * x[j];
            }
            p_y_given_x[i] += b[i];
        }
        
        softmax(p_y_given_x);

        for(int i=0; i<n_out; i++) {
            dy[i] = y[i] - p_y_given_x[i];

            for(int j=0; j<n_in; j++) {
                W[i][j] += lr * dy[i] * x[j] / N;
            }
            b[i] += lr * dy[i] / N;
        }
        return dy;
    }
    
    
    
    //
    public void predict(double[] x, double[] y) {
        for(int i=0; i<n_out; i++) {
            y[i] = 0.;
            for(int j=0; j<n_in; j++) {
                y[i] += W[i][j] * x[j];
            }
            y[i] += b[i];
        }
        softmax(y);
    }
}
