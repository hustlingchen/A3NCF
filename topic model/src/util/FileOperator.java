package util;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;

/**
 * This code is just for simplifying the construction of BufferedRead
 * */
public class FileOperator {

	public BufferedReader read(File file) throws UnsupportedEncodingException,
			FileNotFoundException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(file), "UTF-8"));
		return in;
	}

	public BufferedWriter write(File file) throws UnsupportedEncodingException,
			FileNotFoundException {
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(file), "UTF-8"));
		return bw;
	}
}

