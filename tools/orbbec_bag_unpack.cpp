#include "libobsensor/ObSensor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path input;
    fs::path output;
    uint64_t maxFrames = 0;
    uint64_t stride = 1;
    bool metadataOnly = false;
};

struct ProfileInfo {
    std::string sensor;
    uint32_t index = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
    OBFormat format = OB_FORMAT_UNKNOWN;
};

struct CaptureSummary {
    uint64_t framesetsObserved = 0;
    uint64_t framesetsSaved = 0;
    uint64_t colorFramesSaved = 0;
    uint64_t depthFramesSaved = 0;
    uint64_t irFramesSaved = 0;
    uint64_t bytesWritten = 0;
    float depthValueScaleMm = 0.0f;
};

static void usage(const char *program) {
    std::cerr
        << "Usage: " << program << " --input FILE.bag --output DIR [options]\n"
        << "Options:\n"
        << "  --max-frames N    Save at most N synchronized framesets (0 = all)\n"
        << "  --stride N        Save every Nth frameset (default: 1)\n"
        << "  --metadata-only   Read calibration and stream profiles without playback\n";
}

static Options parseArgs(int argc, char **argv) {
    Options options;
    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string &name) -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if(arg == "--input") {
            options.input = requireValue(arg);
        }
        else if(arg == "--output") {
            options.output = requireValue(arg);
        }
        else if(arg == "--max-frames") {
            options.maxFrames = std::stoull(requireValue(arg));
        }
        else if(arg == "--stride") {
            options.stride = std::stoull(requireValue(arg));
        }
        else if(arg == "--metadata-only") {
            options.metadataOnly = true;
        }
        else if(arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if(options.input.empty() || options.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    if(options.stride == 0) {
        throw std::runtime_error("--stride must be at least 1");
    }
    if(!fs::is_regular_file(options.input)) {
        throw std::runtime_error("Input bag does not exist: " + options.input.string());
    }
    return options;
}

static std::string jsonEscape(const std::string &value) {
    std::ostringstream out;
    for(unsigned char c : value) {
        switch(c) {
        case '\"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if(c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            }
            else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

static std::string formatName(OBFormat format) {
    switch(format) {
    case OB_FORMAT_YUYV: return "YUYV";
    case OB_FORMAT_YUY2: return "YUY2";
    case OB_FORMAT_UYVY: return "UYVY";
    case OB_FORMAT_NV12: return "NV12";
    case OB_FORMAT_NV21: return "NV21";
    case OB_FORMAT_MJPG: return "MJPG";
    case OB_FORMAT_H264: return "H264";
    case OB_FORMAT_H265: return "H265";
    case OB_FORMAT_Y16: return "Y16";
    case OB_FORMAT_Y8: return "Y8";
    case OB_FORMAT_Y10: return "Y10";
    case OB_FORMAT_Y11: return "Y11";
    case OB_FORMAT_Y12: return "Y12";
    case OB_FORMAT_GRAY: return "GRAY";
    case OB_FORMAT_I420: return "I420";
    case OB_FORMAT_RGB: return "RGB";
    case OB_FORMAT_BGR: return "BGR";
    case OB_FORMAT_Y14: return "Y14";
    case OB_FORMAT_BGRA: return "BGRA";
    case OB_FORMAT_Z16: return "Z16";
    case OB_FORMAT_RGBA: return "RGBA";
    case OB_FORMAT_UNKNOWN: return "UNKNOWN";
    default: return "FORMAT_" + std::to_string(static_cast<int>(format));
    }
}

static void appendProfiles(ob::Pipeline &pipe, OBSensorType sensorType, const std::string &name, std::vector<ProfileInfo> &profiles) {
    try {
        const auto list = pipe.getStreamProfileList(sensorType);
        if(!list) {
            return;
        }
        for(uint32_t i = 0; i < list->count(); ++i) {
            const auto base = list->getProfile(i);
            if(!base || !base->is<ob::VideoStreamProfile>()) {
                continue;
            }
            const auto video = base->as<ob::VideoStreamProfile>();
            profiles.push_back({ name, i, video->width(), video->height(), video->fps(), video->format() });
        }
    }
    catch(const ob::Error &) {
        // A recording is allowed to omit any sensor stream.
    }
}

static std::vector<ProfileInfo> collectProfiles(ob::Pipeline &pipe) {
    std::vector<ProfileInfo> profiles;
    appendProfiles(pipe, OB_SENSOR_COLOR, "color", profiles);
    appendProfiles(pipe, OB_SENSOR_DEPTH, "depth", profiles);
    appendProfiles(pipe, OB_SENSOR_IR, "ir", profiles);
    appendProfiles(pipe, OB_SENSOR_IR_LEFT, "ir_left", profiles);
    appendProfiles(pipe, OB_SENSOR_IR_RIGHT, "ir_right", profiles);
    return profiles;
}

static void writeIntrinsic(std::ostream &out, const OBCameraIntrinsic &intrinsic) {
    out << "{\"width\":" << intrinsic.width
        << ",\"height\":" << intrinsic.height
        << ",\"fx\":" << intrinsic.fx
        << ",\"fy\":" << intrinsic.fy
        << ",\"cx\":" << intrinsic.cx
        << ",\"cy\":" << intrinsic.cy << "}";
}

static OBCameraIntrinsic scaleIntrinsic(const OBCameraIntrinsic &source, uint32_t width, uint32_t height) {
    if(source.width <= 0 || source.height <= 0 || width == 0 || height == 0) {
        return source;
    }
    OBCameraIntrinsic scaled = source;
    const float sx = static_cast<float>(width) / source.width;
    const float sy = static_cast<float>(height) / source.height;
    scaled.fx *= sx;
    scaled.fy *= sy;
    scaled.cx *= sx;
    scaled.cy *= sy;
    scaled.width = static_cast<int16_t>(width);
    scaled.height = static_cast<int16_t>(height);
    return scaled;
}

static void writeDistortion(std::ostream &out, const OBCameraDistortion &distortion) {
    out << "{\"k1\":" << distortion.k1
        << ",\"k2\":" << distortion.k2
        << ",\"k3\":" << distortion.k3
        << ",\"k4\":" << distortion.k4
        << ",\"k5\":" << distortion.k5
        << ",\"k6\":" << distortion.k6
        << ",\"p1\":" << distortion.p1
        << ",\"p2\":" << distortion.p2 << "}";
}

static void writeMetadata(const fs::path &path, const Options &options, ob::DeviceInfo &deviceInfo,
                          const OBCameraParam &camera, const std::vector<ProfileInfo> &profiles,
                          const CaptureSummary &summary, bool complete) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) {
        throw std::runtime_error("Cannot write metadata: " + path.string());
    }
    out << std::setprecision(10);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"orbbec_sdk_version\": \"1.10.22\",\n"
        << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
        << "  \"source_bag\": \"" << jsonEscape(fs::absolute(options.input).string()) << "\",\n"
        << "  \"source_size_bytes\": " << fs::file_size(options.input) << ",\n"
        << "  \"device\": {\n"
        << "    \"name\": \"" << jsonEscape(deviceInfo.name()) << "\",\n"
        << "    \"serial_number\": \"" << jsonEscape(deviceInfo.serialNumber()) << "\",\n"
        << "    \"firmware_version\": \"" << jsonEscape(deviceInfo.firmwareVersion()) << "\",\n"
        << "    \"vid\": " << deviceInfo.vid() << ",\n"
        << "    \"pid\": " << deviceInfo.pid() << "\n"
        << "  },\n"
        << "  \"camera\": {\n"
        << "    \"depth_intrinsic\": ";
    writeIntrinsic(out, camera.depthIntrinsic);
    out << ",\n    \"color_intrinsic\": ";
    writeIntrinsic(out, camera.rgbIntrinsic);
    out << ",\n    \"depth_distortion\": ";
    writeDistortion(out, camera.depthDistortion);
    out << ",\n    \"color_distortion\": ";
    writeDistortion(out, camera.rgbDistortion);
    out << ",\n    \"depth_to_color_rotation_row_major\": [";
    for(int i = 0; i < 9; ++i) {
        if(i) out << ',';
        out << camera.transform.rot[i];
    }
    out << "],\n    \"depth_to_color_translation_mm\": [";
    for(int i = 0; i < 3; ++i) {
        if(i) out << ',';
        out << camera.transform.trans[i];
    }
    out << "],\n    \"mirrored\": " << (camera.isMirrored ? "true" : "false") << "\n  },\n";
    out << "  \"stream_profiles\": [\n";
    for(size_t i = 0; i < profiles.size(); ++i) {
        const auto &p = profiles[i];
        out << "    {\"sensor\":\"" << p.sensor << "\",\"profile_index\":" << p.index
            << ",\"width\":" << p.width << ",\"height\":" << p.height
            << ",\"fps\":" << p.fps << ",\"format\":\"" << formatName(p.format) << "\",\"intrinsic\":";
        const bool isColor = p.sensor == "color";
        const OBCameraIntrinsic recordedIntrinsic = isColor ? camera.rgbIntrinsic : camera.depthIntrinsic;
        const OBCameraIntrinsic effectiveIntrinsic = scaleIntrinsic(recordedIntrinsic, p.width, p.height);
        writeIntrinsic(out, effectiveIntrinsic);
        out << ",\"intrinsic_source\":\""
            << ((recordedIntrinsic.width == static_cast<int>(p.width) && recordedIntrinsic.height == static_cast<int>(p.height))
                    ? "recorded_camera_param"
                    : "scaled_from_recorded_camera_param")
            << "\"";
        out << ",\"distortion\":";
        writeDistortion(out, isColor ? camera.rgbDistortion : camera.depthDistortion);
        out << "}";
        out << (i + 1 == profiles.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"export\": {\n"
        << "    \"stride\": " << options.stride << ",\n"
        << "    \"max_frames\": " << options.maxFrames << ",\n"
        << "    \"framesets_observed\": " << summary.framesetsObserved << ",\n"
        << "    \"framesets_saved\": " << summary.framesetsSaved << ",\n"
        << "    \"color_frames_saved\": " << summary.colorFramesSaved << ",\n"
        << "    \"depth_frames_saved\": " << summary.depthFramesSaved << ",\n"
        << "    \"ir_frames_saved\": " << summary.irFramesSaved << ",\n"
        << "    \"bytes_written\": " << summary.bytesWritten << ",\n"
        << "    \"depth_value_scale_mm\": " << summary.depthValueScaleMm << "\n"
        << "  }\n"
        << "}\n";
}

static uint64_t writeBytes(const fs::path &path, const void *data, size_t size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) {
        throw std::runtime_error("Cannot write frame: " + path.string());
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return size;
}

static uint64_t writePgm8(const fs::path &path, ob::VideoFrame &frame) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) {
        throw std::runtime_error("Cannot write frame: " + path.string());
    }
    const std::string header = "P5\n" + std::to_string(frame.width()) + " " + std::to_string(frame.height()) + "\n255\n";
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char *>(frame.data()), static_cast<std::streamsize>(frame.dataSize()));
    return header.size() + frame.dataSize();
}

static uint64_t writePgm16(const fs::path &path, ob::VideoFrame &frame) {
    const size_t pixels = static_cast<size_t>(frame.width()) * frame.height();
    if(frame.dataSize() < pixels * sizeof(uint16_t)) {
        throw std::runtime_error("16-bit frame buffer is smaller than width*height*2");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) {
        throw std::runtime_error("Cannot write frame: " + path.string());
    }
    const std::string header = "P5\n" + std::to_string(frame.width()) + " " + std::to_string(frame.height()) + "\n65535\n";
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const auto *src = static_cast<const uint16_t *>(frame.data());
    std::vector<uint8_t> bigEndian(pixels * 2);
    for(size_t i = 0; i < pixels; ++i) {
        bigEndian[2 * i] = static_cast<uint8_t>(src[i] >> 8);
        bigEndian[2 * i + 1] = static_cast<uint8_t>(src[i] & 0xff);
    }
    out.write(reinterpret_cast<const char *>(bigEndian.data()), static_cast<std::streamsize>(bigEndian.size()));
    return header.size() + bigEndian.size();
}

static uint64_t writePpm(const fs::path &path, ob::VideoFrame &frame, bool sourceIsBgr) {
    const size_t pixels = static_cast<size_t>(frame.width()) * frame.height();
    if(frame.dataSize() < pixels * 3) {
        throw std::runtime_error("RGB frame buffer is smaller than width*height*3");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) {
        throw std::runtime_error("Cannot write frame: " + path.string());
    }
    const std::string header = "P6\n" + std::to_string(frame.width()) + " " + std::to_string(frame.height()) + "\n255\n";
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const auto *src = static_cast<const uint8_t *>(frame.data());
    if(sourceIsBgr) {
        std::vector<uint8_t> rgb(pixels * 3);
        for(size_t i = 0; i < pixels; ++i) {
            rgb[3 * i] = src[3 * i + 2];
            rgb[3 * i + 1] = src[3 * i + 1];
            rgb[3 * i + 2] = src[3 * i];
        }
        out.write(reinterpret_cast<const char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }
    else {
        out.write(reinterpret_cast<const char *>(src), static_cast<std::streamsize>(pixels * 3));
    }
    return header.size() + pixels * 3;
}

static std::string frameStem(uint64_t framesetIndex) {
    std::ostringstream out;
    out << "frame_" << std::setw(8) << std::setfill('0') << framesetIndex;
    return out.str();
}

static fs::path saveVideoFrame(const std::shared_ptr<ob::VideoFrame> &frame, const fs::path &directory,
                               const std::string &stem, uint64_t &bytesWritten) {
    fs::path path;
    switch(frame->format()) {
    case OB_FORMAT_MJPG:
        path = directory / (stem + ".jpg");
        bytesWritten += writeBytes(path, frame->data(), frame->dataSize());
        break;
    case OB_FORMAT_Y8:
    case OB_FORMAT_BA81:
        path = directory / (stem + ".pgm");
        bytesWritten += writePgm8(path, *frame);
        break;
    case OB_FORMAT_Y16:
    case OB_FORMAT_Z16:
    case OB_FORMAT_Y10:
    case OB_FORMAT_Y11:
    case OB_FORMAT_Y12:
    case OB_FORMAT_Y14:
        path = directory / (stem + ".pgm");
        bytesWritten += writePgm16(path, *frame);
        break;
    case OB_FORMAT_RGB:
        path = directory / (stem + ".ppm");
        bytesWritten += writePpm(path, *frame, false);
        break;
    case OB_FORMAT_BGR:
        path = directory / (stem + ".ppm");
        bytesWritten += writePpm(path, *frame, true);
        break;
    case OB_FORMAT_H264:
        path = directory / (stem + ".h264");
        bytesWritten += writeBytes(path, frame->data(), frame->dataSize());
        break;
    case OB_FORMAT_H265:
        path = directory / (stem + ".h265");
        bytesWritten += writeBytes(path, frame->data(), frame->dataSize());
        break;
    default:
        path = directory / (stem + ".bin");
        bytesWritten += writeBytes(path, frame->data(), frame->dataSize());
        break;
    }
    return path;
}

static void writeManifestRow(std::ofstream &manifest, uint64_t framesetIndex, const std::string &stream,
                             const std::shared_ptr<ob::VideoFrame> &frame, const fs::path &relativePath,
                             float depthScaleMm) {
    manifest << framesetIndex << ',' << stream << ',' << frame->index() << ','
             << frame->timeStampUs() << ',' << frame->systemTimeStampUs() << ',' << frame->globalTimeStampUs() << ','
             << frame->width() << ',' << frame->height() << ',' << formatName(frame->format()) << ','
             << frame->dataSize() << ',' << depthScaleMm << ',' << relativePath.generic_string() << '\n';
    manifest.flush();
}

static CaptureSummary unpack(ob::Pipeline &pipe, const Options &options, ob::DeviceInfo &deviceInfo,
                             const OBCameraParam &camera, const std::vector<ProfileInfo> &profiles) {
    fs::create_directories(options.output / "color");
    fs::create_directories(options.output / "depth");
    fs::create_directories(options.output / "ir");

    std::ofstream manifest(options.output / "frames.csv", std::ios::binary | std::ios::trunc);
    if(!manifest) {
        throw std::runtime_error("Cannot write frames.csv");
    }
    manifest << "frameset_index,stream,frame_index,device_timestamp_us,system_timestamp_us,global_timestamp_us,"
                "width,height,format,data_size_bytes,depth_value_scale_mm,file\n";

    CaptureSummary summary;
    writeMetadata(options.output / "metadata.json", options, deviceInfo, camera, profiles, summary, false);

    std::atomic<bool> ended{ false };
    const auto playback = pipe.getPlayback();
    playback->setPlaybackStateCallback([&](OBMediaState state) {
        if(state == OB_MEDIA_END) {
            ended.store(true);
        }
    });

    pipe.start(nullptr);
    uint32_t consecutiveTimeouts = 0;
    while(!ended.load()) {
        const auto frameset = pipe.waitForFrames(1000);
        if(!frameset) {
            if(++consecutiveTimeouts >= 30) {
                throw std::runtime_error("Timed out waiting 30 seconds for playback frames");
            }
            continue;
        }
        consecutiveTimeouts = 0;
        const uint64_t framesetIndex = summary.framesetsObserved++;
        if(framesetIndex % options.stride != 0) {
            continue;
        }

        const std::string stem = frameStem(framesetIndex);
        if(const auto color = frameset->colorFrame()) {
            const auto path = saveVideoFrame(color, options.output / "color", stem, summary.bytesWritten);
            writeManifestRow(manifest, framesetIndex, "color", color, fs::relative(path, options.output), 0.0f);
            ++summary.colorFramesSaved;
        }
        if(const auto depth = frameset->depthFrame()) {
            summary.depthValueScaleMm = depth->getValueScale();
            const auto path = saveVideoFrame(depth, options.output / "depth", stem, summary.bytesWritten);
            writeManifestRow(manifest, framesetIndex, "depth", depth, fs::relative(path, options.output), summary.depthValueScaleMm);
            ++summary.depthFramesSaved;
        }
        if(const auto ir = frameset->irFrame()) {
            const auto path = saveVideoFrame(ir, options.output / "ir", stem, summary.bytesWritten);
            writeManifestRow(manifest, framesetIndex, "ir", ir, fs::relative(path, options.output), 0.0f);
            ++summary.irFramesSaved;
        }
        ++summary.framesetsSaved;

        if(summary.framesetsSaved % 100 == 0 || (options.maxFrames > 0 && summary.framesetsSaved == options.maxFrames)) {
            std::cout << "saved_framesets=" << summary.framesetsSaved
                      << " observed_framesets=" << summary.framesetsObserved
                      << " bytes_written=" << summary.bytesWritten << std::endl;
            writeMetadata(options.output / "metadata.json", options, deviceInfo, camera, profiles, summary, false);
        }
        if(options.maxFrames > 0 && summary.framesetsSaved >= options.maxFrames) {
            break;
        }
    }
    pipe.stop();
    writeMetadata(options.output / "metadata.json", options, deviceInfo, camera, profiles, summary, true);
    return summary;
}

int main(int argc, char **argv) try {
    const Options options = parseArgs(argc, argv);
    fs::create_directories(options.output);

    ob::Pipeline pipe(options.input.string().c_str());
    const auto playback = pipe.getPlayback();
    const auto deviceInfo = playback->getDeviceInfo();
    const auto profiles = collectProfiles(pipe);
    OBCameraParam camera = pipe.getCameraParam();
    uint32_t colorWidth = 0, colorHeight = 0, depthWidth = 0, depthHeight = 0;
    for(const auto &profile : profiles) {
        if(profile.sensor == "color") {
            colorWidth = profile.width;
            colorHeight = profile.height;
        }
        else if(profile.sensor == "depth") {
            depthWidth = profile.width;
            depthHeight = profile.height;
        }
    }
    if(colorWidth > 0 && colorHeight > 0 && depthWidth > 0 && depthHeight > 0) {
        camera = pipe.getCameraParamWithProfile(colorWidth, colorHeight, depthWidth, depthHeight);
    }

    std::cout << "device=" << deviceInfo->name()
              << " serial=" << deviceInfo->serialNumber()
              << " firmware=" << deviceInfo->firmwareVersion() << '\n';
    for(const auto &profile : profiles) {
        std::cout << "profile sensor=" << profile.sensor << " index=" << profile.index
                  << " size=" << profile.width << 'x' << profile.height
                  << " fps=" << profile.fps << " format=" << formatName(profile.format) << '\n';
    }

    CaptureSummary summary;
    if(options.metadataOnly) {
        writeMetadata(options.output / "metadata.json", options, *deviceInfo, camera, profiles, summary, true);
    }
    else {
        summary = unpack(pipe, options, *deviceInfo, camera, profiles);
    }

    std::cout << "complete output=" << fs::absolute(options.output)
              << " framesets_saved=" << summary.framesetsSaved
              << " bytes_written=" << summary.bytesWritten << std::endl;
    return 0;
}
catch(const ob::Error &error) {
    std::cerr << "Orbbec error: function=" << error.getName()
              << " args=" << error.getArgs()
              << " message=" << error.getMessage()
              << " type=" << error.getExceptionType() << std::endl;
    return 2;
}
catch(const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
}
