#include <iostream>
#include "cache_file.h"

using namespace std;
namespace nervana
{
	CacheFile::CacheFile(const std::string& file_name)
	{
		file_i_.open(file_name, ios::in|ios::binary);
		if (!file_i_.is_open())
			throw std::runtime_error("can't open cache file");

		uint32_t check_file_ID;
		file_i_.read(reinterpret_cast<char*>(&check_file_ID), sizeof(uint32_t));
		if (check_file_ID != file_ID)
			throw std::runtime_error("wrong file ID");
		
		uint64_t records_number;
		file_i_.read(reinterpret_cast<char*>(&records_number), sizeof(uint64_t));
		cout<<"loading "<<records_number<<" items\n";
		offests_header_.resize(records_number);
		file_i_.read(reinterpret_cast<char*>(offests_header_.data()), offests_header_.size()*sizeof(uint64_t));	
		random_.seed(1); // !!!!!!!!!!!!!!
		std::shuffle(offests_header_.begin(), offests_header_.end(), random_);
		current_id = 0;
	}

	encoded_record CacheFile::get_record()
	{
		if (current_id >= offests_header_.size())
		{
			std::shuffle(offests_header_.begin(), offests_header_.end(), random_);
			current_id = 0;
		}
		file_i_.seekg(offests_header_[current_id]);
		current_id++;

		uint32_t check_record_ID;
		file_i_.read(reinterpret_cast<char*>(&check_record_ID), sizeof(uint32_t));
		if (check_record_ID != record_ID)
			throw std::runtime_error("wrong record ID");

		uint64_t record_size;
		file_i_.read(reinterpret_cast<char*>(&record_size), sizeof(uint64_t));
		encoded_record record;
		for (int i = 0; i < record_size; i++)
		{
			uint64_t size;
			file_i_.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
			std::vector<char> record_buf;
			//cout<<size<<"\n";
			record_buf.resize(size);
			file_i_.read(record_buf.data(), size);
			record.add_element(std::move(record_buf));
		}
		return record;
	}

	CacheFile::CacheFile(const std::string& file_name, uint64_t records_number)
	{
		mode_read =false;
		current_id = 0;
		offests_header_.resize(records_number);
		file_.open(file_name, ios::binary | ios::out);
		if (file_.is_open())
			cout<<"open Ok\n";
		else
			cout<<"fail!!!!!!\n";
			
		// check
		file_.write(reinterpret_cast<const char*>(&file_ID), sizeof(uint32_t));	
		file_.write(reinterpret_cast<char*>(&records_number), sizeof(uint64_t));	
		file_.write(reinterpret_cast<char*>(offests_header_.data()), offests_header_.size()*sizeof(uint64_t));
		current_id = 0;	
	}
	


	void CacheFile::add_record(encoded_record& record)
	{
		auto pos = file_.tellp();
		offests_header_[current_id] = pos;
		uint64_t record_size = record.size();
		file_.write(reinterpret_cast<const char*>(&record_ID), sizeof(uint32_t));
		file_.write(reinterpret_cast<char*>(&record_size), sizeof(uint64_t));
		for (auto& element: record)
		{
			uint64_t size = element.size();
			file_.write(reinterpret_cast<char*>(&size), sizeof(uint64_t));
			file_.write(element.data(), size);
		}
		current_id++;
	}
	CacheFile::~CacheFile()
	{
		if (!mode_read)
		{
			file_.seekp(sizeof(uint32_t)+sizeof(uint64_t));
			cout<<"write offset table\n";
			file_.write(reinterpret_cast<char*>(offests_header_.data()), offests_header_.size()*sizeof(uint64_t));	
			cout<<"flush file\n";
			file_.close();
			cout<<"file writed\n";
		}
		else
		{
			file_i_.close();
		}
	}



}