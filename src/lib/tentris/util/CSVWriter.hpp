#include <fstream>
#include <string>
#include <vector>

class CSVWriter {
	std::string fileName;
	std::string delimeter;
	std::ofstream file;
	bool isFileEmpty;

public:
	CSVWriter(std::string filename, std::string delm = ",") :
															  fileName(filename), delimeter(delm), file(filename, std::ios::out | std::ios::app) {
		// Check if file exists and if it is empty
		std::ifstream inFile(fileName);
		isFileEmpty = inFile.peek() == std::ifstream::traits_type::eof();
	}

	void writeRow(const std::vector<std::string>& row) {
		if (file.is_open()) {
			if (isFileEmpty) {
				// The file was empty, we write the header
				for (size_t i = 0; i < row.size(); ++i) {
					file << row[i];
					if (i != row.size() - 1) file << delimeter;
				}
				file << std::endl;
				isFileEmpty = false; // Update the flag as the file is no longer empty
			} else {
				// The file was not empty, we write the data
				for (size_t i = 0; i < row.size(); ++i) {
					file << row[i];
					if (i != row.size() - 1) file << delimeter;
				}
				file << std::endl;
			}
		}
	}

	~CSVWriter() {
		if (file.is_open()) {
			file.close();
		}
	}
};
