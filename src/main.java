import java.io.*;
import java.util.ArrayList;

public class main {

    private static ArrayList<ArrayList<Double[]>> test_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> test_desired_data = new ArrayList<>();

    private static ArrayList<ArrayList<Double[]>> train_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> train_desired_data = new ArrayList<>();

    //TODO
    private static int NumberOftest = 10;

    public static void main(String[] args) throws IOException {

        for(int tain_i = 0 ; tain_i < NumberOftest ; tain_i ++) {

            ArrayList<Double[]> test_dataset_i = new ArrayList<>();
            ArrayList<Double[]> test_desired_data_i = new ArrayList<>();

            ArrayList<Double[]> train_dataset_i = new ArrayList<>();
            ArrayList<Double[]> train_desired_data_i = new ArrayList<>();


            FileInputStream fstream = new FileInputStream("src/Flood_dataset.txt");
            DataInputStream in = new DataInputStream(fstream);
            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String data;

            int line_i = 0;
            while ((data = br.readLine()) != null) { // each line
                String[] tmp = data.split("\t");    //Split space

                Double[] tmp_dataset = new Double[tmp.length-1];
                Double[] tmp_desired_data = new Double[1];

                int word_i = 0;
                for (String t : tmp) {  // each word
                    double tmp_val = (Double.parseDouble(t)/700.0);  // norm

                        if (word_i == tmp.length - 1) { //  desired_data
                            tmp_desired_data[0] = (tmp_val) ;
                        } else {
                            tmp_dataset[word_i] = (tmp_val)  ;
                        }

                    word_i++;
                }

                if (line_i % (10) == tain_i) { // test 10% data
                    test_dataset_i.add(tmp_dataset);
                    test_desired_data_i.add(tmp_desired_data);
                } else { //train 90%
                    train_dataset_i.add(tmp_dataset);
                    train_desired_data_i.add(tmp_desired_data);
                }
                line_i++;
            }
            test_dataset.add(test_dataset_i);
            test_desired_data.add(test_desired_data_i);
            train_dataset.add(train_dataset_i);
            train_desired_data.add(train_desired_data_i);

        }


        for(int test_i = 1 ; test_i < NumberOftest ; test_i ++) {
//            int test_i = 0;
            System.out.println("===================================================");
            brain b1 = new brain("8,5,5,5,1", 5000, 0.00001, 1, 0.05, 0.9);
            System.out.println("train: " + test_i);
            b1.train( train_dataset.get(test_i), train_desired_data.get(test_i));
            System.out.println("test: " + test_i);
            b1.test(test_dataset.get(test_i), test_desired_data.get(test_i));
            System.out.println("===================================================");
        }
    }

}
