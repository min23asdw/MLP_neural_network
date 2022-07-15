

import java.io.*;
import java.util.ArrayList;

public class main {

    public static void main(String[] args) throws IOException {
        ArrayList<Double[]> dataset = new ArrayList<>();
        ArrayList<Double[]> desired_data = new ArrayList<>();


        FileInputStream fstream = new FileInputStream("src/Flood_dataset.txt");
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String data;
        int line =0;
        while ((data = br.readLine()) != null)   {
            String[] tmp =data.split("\t");    //Split space
            Double[] tmp_data = new Double[tmp.length];
            Double[] temp_desired = new Double[1];
            int i =0;
            for (String t: tmp) {
                if(i==tmp.length-1){
                    temp_desired[0] = Double.parseDouble(t);
                }else{
                    tmp_data[i] = Double.parseDouble(t);
                }

                i++;
            }


            dataset.add(tmp_data);
            desired_data.add(temp_desired);
        }
        brain b1 = new brain("4332", dataset ,desired_data ,2000,0.000001);

        b1.train();
    }
}
