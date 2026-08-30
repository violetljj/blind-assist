#include <XnCppWrapper.h>
#include <XnOpenNI.h>

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

typedef unsigned char BYTE;
struct FIBITMAP;
enum FREE_IMAGE_FORMAT { FIF_UNKNOWN = -1, FIF_PNG = 13 };
enum FREE_IMAGE_TYPE { FIT_UNKNOWN = 0, FIT_BITMAP = 1, FIT_UINT16 = 2 };

extern "C" {
__declspec(dllimport) FIBITMAP* FreeImage_Allocate(
    int width, int height, int bpp, unsigned red_mask, unsigned green_mask,
    unsigned blue_mask);
__declspec(dllimport) FIBITMAP* FreeImage_AllocateT(
    FREE_IMAGE_TYPE type, int width, int height, int bpp, unsigned red_mask,
    unsigned green_mask, unsigned blue_mask);
__declspec(dllimport) unsigned FreeImage_GetLine(FIBITMAP* dib);
__declspec(dllimport) unsigned FreeImage_GetWidth(FIBITMAP* dib);
__declspec(dllimport) BYTE* FreeImage_GetBits(FIBITMAP* dib);
__declspec(dllimport) unsigned FreeImage_GetPitch(FIBITMAP* dib);
__declspec(dllimport) BYTE* FreeImage_GetScanLine(FIBITMAP* dib, int scanline);
__declspec(dllimport) void FreeImage_FlipVertical(FIBITMAP* dib);
__declspec(dllimport) int FreeImage_Save(
    FREE_IMAGE_FORMAT fif, FIBITMAP* dib, const char* filename, int flags);
__declspec(dllimport) void FreeImage_Unload(FIBITMAP* dib);
}

namespace {

constexpr long long kTimeDiff = 33000;
constexpr int kRgbaBlue = 0;
constexpr int kRgbaGreen = 1;
constexpr int kRgbaRed = 2;

bool make_directory(const std::string& path) {
    if (CreateDirectoryA(path.c_str(), nullptr)) {
        return true;
    }
    return GetLastError() == ERROR_ALREADY_EXISTS;
}

bool check_status(XnStatus status, const char* operation) {
    if (status == XN_STATUS_OK) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, xnGetStatusString(status));
    return false;
}

void color_to_bitmap(xn::ImageMetaData& color_metadata, FIBITMAP* bitmap) {
    xn::RGB24Map& color_map = color_metadata.WritableRGB24Map();
    const int bytes_per_pixel =
        static_cast<int>(FreeImage_GetLine(bitmap) / FreeImage_GetWidth(bitmap));
    BYTE* bits = FreeImage_GetBits(bitmap);
    const unsigned pitch = FreeImage_GetPitch(bitmap);
    for (XnUInt32 y = 0; y < color_map.YRes(); ++y) {
        BYTE* pixel = bits;
        for (XnUInt32 x = 0; x < color_map.XRes(); ++x) {
            pixel[kRgbaRed] = color_map(x, y).nRed;
            pixel[kRgbaGreen] = color_map(x, y).nGreen;
            pixel[kRgbaBlue] = color_map(x, y).nBlue;
            pixel += bytes_per_pixel;
        }
        bits += pitch;
    }
    FreeImage_FlipVertical(bitmap);
}

void depth_to_bitmap(xn::DepthMetaData& depth_metadata, FIBITMAP* bitmap) {
    xn::DepthMap& depth_map = depth_metadata.WritableDepthMap();
    for (XnUInt32 y = 0; y < depth_map.YRes(); ++y) {
        auto* bits = reinterpret_cast<unsigned short*>(FreeImage_GetScanLine(bitmap, y));
        for (XnUInt32 x = 0; x < depth_map.XRes(); ++x) {
            bits[x] = depth_map(x, y);
        }
    }
    FreeImage_FlipVertical(bitmap);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::fprintf(
            stderr,
            "Usage: %s <ONI file> <output folder> <trajectory frame> <trajectory frame> [...]\n",
            argv[0]);
        return 2;
    }
    const std::string input_file = argv[1];
    const std::string output_root = argv[2];
    std::set<int> targets;
    for (int index = 3; index < argc; ++index) {
        char* end = nullptr;
        const long value = std::strtol(argv[index], &end, 10);
        if (end == argv[index] || *end != '\0' || value < 0) {
            std::fprintf(stderr, "Invalid trajectory frame: %s\n", argv[index]);
            return 2;
        }
        targets.insert(static_cast<int>(value));
    }
    if (targets.empty()) {
        return 2;
    }
    const int maximum_target = *targets.rbegin();
    if (!make_directory(output_root) ||
        !make_directory(output_root + "/image") ||
        !make_directory(output_root + "/depth")) {
        std::fprintf(stderr, "Cannot create output directories: %s\n", output_root.c_str());
        return 3;
    }

    xn::Context context;
    if (!check_status(context.Init(), "Context.Init")) {
        return 4;
    }
    xn::Player player;
    if (!check_status(context.OpenFileRecording(input_file.c_str(), player),
                      "Context.OpenFileRecording") ||
        !check_status(player.SetRepeat(false), "Player.SetRepeat") ||
        !check_status(player.SetPlaybackSpeed(0), "Player.SetPlaybackSpeed")) {
        return 4;
    }
    xn::DepthGenerator depth_stream;
    xn::ImageGenerator color_stream;
    if (!check_status(context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth_stream),
                      "Find depth stream") ||
        !check_status(context.FindExistingNode(XN_NODE_TYPE_IMAGE, color_stream),
                      "Find color stream")) {
        return 4;
    }

    FIBITMAP* color_bitmap = FreeImage_Allocate(640, 480, 24, 0, 0, 0);
    FIBITMAP* depth_bitmap = FreeImage_AllocateT(FIT_UINT16, 640, 480, 16, 0, 0, 0);
    if (color_bitmap == nullptr || depth_bitmap == nullptr) {
        std::fprintf(stderr, "FreeImage allocation failed\n");
        return 5;
    }
    const std::string timestamp_path = output_root + "/selected_timestamp.txt";
    std::FILE* timestamp_file = std::fopen(timestamp_path.c_str(), "w");
    if (timestamp_file == nullptr) {
        std::fprintf(stderr, "Cannot open %s\n", timestamp_path.c_str());
        return 5;
    }

    int synchronized_index = 0;
    int discarded_depth = 0;
    int discarded_color = 0;
    int saved = 0;
    bool reached_eof = false;
    while (!reached_eof) {
        xn::DepthMetaData depth_metadata;
        xn::ImageMetaData color_metadata;
        if (depth_stream.WaitAndUpdateData() == XN_STATUS_EOF) {
            break;
        }
        depth_stream.GetMetaData(depth_metadata);
        if (color_stream.WaitAndUpdateData() == XN_STATUS_EOF) {
            break;
        }
        color_stream.GetMetaData(color_metadata);

        XnUInt64 color_timestamp = color_metadata.Timestamp();
        XnUInt64 depth_timestamp = depth_metadata.Timestamp();
        long long difference = std::llabs(
            static_cast<long long>(color_timestamp) -
            static_cast<long long>(depth_timestamp));
        while (difference > kTimeDiff) {
            if (color_timestamp > depth_timestamp) {
                ++discarded_depth;
                const XnStatus status = depth_stream.WaitAndUpdateData();
                if (status == XN_STATUS_EOF) {
                    reached_eof = true;
                    break;
                }
                if (!check_status(status, "Advance depth stream")) {
                    return 6;
                }
                depth_stream.GetMetaData(depth_metadata);
            } else {
                ++discarded_color;
                const XnStatus status = color_stream.WaitAndUpdateData();
                if (status == XN_STATUS_EOF) {
                    reached_eof = true;
                    break;
                }
                if (!check_status(status, "Advance color stream")) {
                    return 6;
                }
                color_stream.GetMetaData(color_metadata);
            }
            color_timestamp = color_metadata.Timestamp();
            depth_timestamp = depth_metadata.Timestamp();
            difference = std::llabs(
                static_cast<long long>(color_timestamp) -
                static_cast<long long>(depth_timestamp));
        }
        if (reached_eof) {
            break;
        }

        ++synchronized_index;
        const int trajectory_frame = synchronized_index - 1;
        if (targets.count(trajectory_frame) != 0) {
            color_to_bitmap(color_metadata, color_bitmap);
            depth_to_bitmap(depth_metadata, depth_bitmap);
            char leaf_name[32];
            std::snprintf(leaf_name, sizeof(leaf_name), "frame.%04d.png",
                          trajectory_frame);
            const std::string image_name = output_root + "/image/" + leaf_name;
            const std::string depth_name = output_root + "/depth/" + leaf_name;
            if (!FreeImage_Save(FIF_PNG, color_bitmap, image_name.c_str(), 0) ||
                !FreeImage_Save(FIF_PNG, depth_bitmap, depth_name.c_str(), 0)) {
                std::fprintf(stderr, "FreeImage save failed at frame %d\n", trajectory_frame);
                return 7;
            }
            std::fprintf(
                timestamp_file, "%d %d %llu %llu %u %u\n", trajectory_frame,
                synchronized_index,
                static_cast<unsigned long long>(color_metadata.Timestamp()),
                static_cast<unsigned long long>(depth_metadata.Timestamp()),
                color_metadata.FrameID(), depth_metadata.FrameID());
            std::fflush(timestamp_file);
            ++saved;
        }
        if (trajectory_frame >= maximum_target) {
            break;
        }
    }
    std::fclose(timestamp_file);
    FreeImage_Unload(color_bitmap);
    FreeImage_Unload(depth_bitmap);

    const std::string summary_path = output_root + "/summary.txt";
    std::FILE* summary = std::fopen(summary_path.c_str(), "w");
    if (summary == nullptr) {
        return 8;
    }
    std::fprintf(summary,
                 "scanned=%d discarded_depth=%d discarded_color=%d requested=%zu saved=%d\n",
                 synchronized_index, discarded_depth, discarded_color, targets.size(), saved);
    std::fclose(summary);
    std::printf("Scanned %d synchronized frames; saved %d/%zu selected frames.\n",
                synchronized_index, saved, targets.size());
    return saved == static_cast<int>(targets.size()) ? 0 : 9;
}
