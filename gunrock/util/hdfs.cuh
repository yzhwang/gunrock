/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 *  Modifications Copyright (C) 2019 Gunrock. 
 */

// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hdfs.cuh
 *
 * @brief Hadoop File System support
 */

#pragma once

// Requires the hdfs library
#ifdef HAS_HADOOP
extern "C" {
  #include <hdfs.h>
}
#endif

#include <assert.h>
#include <vector>
#include <boost/iostreams/stream.hpp>
#include <gunrock/util/test_utils.cuh>

namespace gunrock {
namespace util {

#ifdef HAS_HADOOP
// define hadoop file IO wrapper
class Hdfs {
private:
  /** the primary filesystem object */
  hdfsFS filesystem;
public:
  /** hdfs file source is used to construct boost iostreams */
  class HdfsDevice {
  private:
    hdfsFS filesystem;
    hdfsFile file;

  public:
    // boost iostream concepts
    typedef char    char_type;
    struct category :
      public boost::iostreams::bidirectional_device_tag,
      public boost::iostreams::multichar_tag,
      public boost::iostreams::closable_tag { };
    
    HdfsDevice() : filesystem(nullptr), file(nullptr) { }

    HdfsDevice(const Hdfs& hdfs_fs, const std::string& filename,
               const bool write = false) :
      filesystem(hdfs_fs.filesystem) {
      assert(filesystem != nullptr);
      // open the file
      const int flags = write ? O_WRONLY : O_RDONLY;
      const int buffer_size = 0; // use default
      const short replication = 0; // use default
      const tSize block_size = 0; // use default
      file = hdfsOpenFile(filesystem, filename.c_str(), flags, buffer_size,
                          replication, block_size);
    }

    void close(std::ios_base::openmode mode = std::ios_base::openmode()) {
      if (file == nullptr) return;
      if (file->type == OUTPUT) {
        const int flush_error = hdfsFlush(filesystem, file);
        assert(flush_error, 0);
      }
      const int close_error = hdfsCloseFile(filesystem, file);
      assert(close_error, 0);
      file = nullptr;
    }

    /** the optimal buffer size is 0. */
    inline std::streamsize optimal_buffer_size() const { return 0; }

    std::streamsize read(char* strm_ptr, std::streamsize n) {
      return hdfsRead(filesystem, file, strm_ptr, n);
    } // end of read
    std::streamsize write(const char* strm_ptr, std::streamsize n) {
      return hdfsWrite(filesystem, file, strm_ptr, n);
    }
    bool good() const { return file != nullptr; }
  }; // end of HdfsDevice

  typedef boost::iostreams::stream<HdfsDevice> fstream;

  // Open a connection to the filesystem. The default arguments
  // should be sufficient for most uses
  Hdfs(const std::string& host = "default", tPort port = 0) {
    filesystem = hdfsConnect(host.c_str(), port);
    assert(filesystem != nullptr);
  } // end of constructor

  ~Hdfs() {
    const int error = hdfsDisconnect(filesystem);
    assert(error, 0);
  } // end of destructor

  inline std::vector<std::string> list_files(const std::string& path) {
    int num_files = 0;
    hdfsFileInfo* hdfs_file_list_ptr =
      hdfsListDirectory(filesystem, path.c_str(), &num_files);
    // copy the file list to the string array
    std::vector<std::string> files(num_files);
    for (int i = 0; i < num_files; ++i) {
      files[i] = std::string(hdfs_file_list_ptr[i].mName);
    }
    // free the file list pointer
    hdfsFreeFileInfo(hdfs_file_list_ptr, num_files);
    return files;
  } // end of list_files

  inline static bool has_hadoop() { return true; }

  static hdfs& get_hdfs();
}; // end of class hdfs
#else
  
class Hdfs {
public:
  /** hdfs file source is used to construct boost iostreams */
  class HdfsDevice {
  public: // boost iostream concepts
    typedef char                                        char_type;
    typedef boost::iostreams::bidirectional_device_tag  category;

  public:
    HdfsDevice(const Hdfs& hdfs_fs, const std::string& filename,
               const bool write = false) {
      util::GRError("Libhdfs is not installed on this system.",
                    __FILE__, __LINE__);
    }

    void close() { }

    std::streamsize read(char* strm_ptr, std::streamsize n) {
      util::GRError("Libhdfs is not installed on this system.",
                    __FILE__, __LINE__);
      return 0;
    } // end of read()

    std::streamsize write(const char* strm_ptr, std::streamsize n) {
      util::GRError("Libhdfs is not installed on this system.",
                    __FILE__, __LINE__);
      return 0;
    }

    bool good() const { return false; }
  }; // end of hdfs device

  typedef boost::iostreams::stream<HdfsDevice> fstream;

  Hdfs(const std::string& host = "default", int port = 0) {
    util::GRError("Libhdfs is not installed on this system.",
                    __FILE__, __LINE__);
  } // end of constructor

  inline std::vector<std::string> list_files(const std::string& path) {
    util::GRError("Libhdfs is not installed on this system.",
                    __FILE__, __LINE__);
      return std::vector<std::string>();
  } // end of list_files

  // No hadoop available
  inline static bool has_hadoop() { return false; }

  static Hdfs& get_hdfs();
}; // end of class Hdfs
#endif

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: