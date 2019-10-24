#pragma once

#include <vector>
#include <fstream>
#include <string>

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "async_manager.hpp"
#include "buffer_batch.hpp"

namespace nervana
{

	class CacheFile
	{
	public:
		CacheFile(const std::string& file_name, uint64_t records_number);
		CacheFile(const std::string& file_name);
		~CacheFile();
		void add_record(encoded_record& record);
		encoded_record get_record();
		size_t get_record_count() const { return offests_header_.size();}
		size_t get_elements_per_record() const { return 2;} //!!!!!!!!!!!!!!s
	protected:
		const uint32_t file_ID = 0xAECD;
		const uint32_t record_ID = 0xAE4D;
		std::vector<uint64_t> offests_header_;
		std::ofstream file_;
		std::ifstream file_i_;
		uint64_t current_id;
		bool mode_read = true;
		random_engine_t    random_;

		int fd_;
		size_t map_size_;
		char* map_data_;
	};

	class CacheSource: public nervana::async_manager_source<encoded_record>, public CacheFile
	{
		public:
		CacheSource(const std::string& file_name):CacheFile(file_name){}
		virtual encoded_record* next()  {return nullptr;}
    	virtual size_t  record_count() const {return get_record_count();}
    	virtual size_t  elements_per_record() const {return get_elements_per_record();}
    	virtual void    reset() {current_id = 0;};

	};
}