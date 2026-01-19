#include "libspu/kernel/hal/logstar.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

class DuplicateBrentKungTest : public ::testing::Test {};
class ExtractOrderedTest : public ::testing::Test {};
namespace {
SPUContext makeSPUContextWithProfile(
    ProtocolKind prot_kind, FieldType field,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig cfg;
  cfg.protocol = prot_kind;
  cfg.field = field;
  cfg.enable_action_trace = false;

  if (lctx->Rank() == 0) {
    cfg.enable_hal_profile = true;
    cfg.enable_pphlo_profile = true;
  }
  return test::makeSPUContext(cfg, lctx);
}
}  // namespace

TEST_F(DuplicateBrentKungTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    int64_t n = 8;
    int64_t block_size = 2;
    xt::xarray<float> x = {5, 6, 2, 2, 10, 10, 4, 3, 1, 1, 2, 2, 5, 5, 3, 2};
    xt::xarray<float> valids = {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    xt::xarray<float> g = {0, 1, 1, 0, 1, 1, 1, 0};
    xt::xarray<float> x_out_expected = {{5, 6}, {5, 6}, {5, 6}, {4, 3},
                                        {4, 3}, {4, 3}, {4, 3}, {3, 2}};
    xt::xarray<float> valid_out_expected = {{1, 1}, {1, 1}, {1, 1}, {0, 1},
                                            {0, 1}, {0, 1}, {0, 1}, {1, 1}};

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    auto x_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, valid_out_s));
    if (lctx->Rank() == 0) {
      std::cout << "x_out_expected:\n" << x_out_expected << std::endl;
      std::cout << "x_out_opened (Actual):\n" << x_out_opened << std::endl;
      std::cout << "valid_out_expected:\n" << valid_out_expected << std::endl;
      std::cout << "valid_out_opened (Actual):\n"
                << valid_out_opened << std::endl;
    }
    EXPECT_TRUE(xt::allclose(x_out_opened, x_out_expected));
    EXPECT_EQ(valid_out_opened.shape()[0], n);
    EXPECT_EQ(valid_out_opened.shape()[1], block_size);
    EXPECT_TRUE(xt::allclose(valid_out_opened, valid_out_expected));
  });
}

TEST_F(DuplicateBrentKungTest, LargeScaleRealNumbers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    const int64_t n = 1000000;
    const int64_t block_size = 2;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist_x(0, 1000);
    std::uniform_int_distribution<int> dist_binary(0, 1);  // 改为 int 类型

    xt::xarray<float> x = xt::zeros<float>({n * block_size});
    xt::xarray<float> valids = xt::zeros<float>({n * block_size});
    xt::xarray<float> g = xt::zeros<float>({n});

    for (auto& v : x) v = dist_x(rng);
    for (auto& v : valids) v = static_cast<float>(dist_binary(rng));
    g[0] = 0.0F;
    for (int64_t i = 1; i < n; ++i) {
      g[i] = static_cast<float>(dist_binary(rng));
    }

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    if (lctx->Rank() == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "duplicate_brent_kung Large Scale Test (Real numbers):"
                << std::endl;
      std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    auto x_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, valid_out_s));

    // verifiy correctness
    if (lctx->Rank() == 0) {
      const float atol = 0.01;
      for (int64_t i = 0; i < n; ++i) {
        if (g(i, 0) == 0) {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x(i, b), atol)
                << "x reset failed at i=" << i << ", b=" << b
                << ", diff=" << std::abs(x_out_opened(i, b) - x(i, b));
            EXPECT_NEAR(valid_out_opened(i, b), valids(i, b), atol)
                << "valids reset failed at i=" << i << ", b=" << b;
          }
        } else {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x_out_opened(i - 1, b), atol)
                << "x propagation failed at i=" << i << ", b=" << b
                << ", diff = "
                << std::abs(x_out_opened(i, b) - x_out_opened(i - 1, b));
            EXPECT_NEAR(valid_out_opened(i, b), valid_out_opened(i - 1, b),
                        atol)
                << "valids propagation failed at i=" << i << ", b=" << b;
          }
        }
      }
    }
  });
}

TEST_F(DuplicateBrentKungTest, LargeScaleIntergers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    const int64_t n = 1000000;
    const int64_t block_size = 2;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_x(0, 1000);
    std::uniform_int_distribution<int> dist_binary(0, 1);

    xt::xarray<int> x = xt::zeros<int>({n * block_size});
    xt::xarray<int> valids = xt::zeros<int>({n * block_size});
    xt::xarray<int> g = xt::zeros<int>({n});

    for (auto& v : x) v = dist_x(rng);
    for (auto& v : valids) v = dist_binary(rng);
    g[0] = 0;
    for (int64_t i = 1; i < n; ++i) {
      g[i] = dist_binary(rng);
    }

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    if (lctx->Rank() == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "duplicate_brent_kung Large Scale Test (Intergers):"
                << std::endl;
      std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    auto x_out_opened =
        hal::dump_public_as<int>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<int>(&ctx, hal::reveal(&ctx, valid_out_s));

    // verifiy correctness
    if (lctx->Rank() == 0) {
      const float atol = 0.01;
      for (int64_t i = 0; i < n; ++i) {
        if (g(i, 0) == 0) {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x(i, b), atol)
                << "x reset failed at i=" << i << ", b=" << b
                << ", diff=" << std::abs(x_out_opened(i, b) - x(i, b));
            EXPECT_NEAR(valid_out_opened(i, b), valids(i, b), atol)
                << "valids reset failed at i=" << i << ", b=" << b;
          }
        } else {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x_out_opened(i - 1, b), atol)
                << "x propagation failed at i=" << i << ", b=" << b
                << ", diff = "
                << std::abs(x_out_opened(i, b) - x_out_opened(i - 1, b));
            EXPECT_NEAR(valid_out_opened(i, b), valid_out_opened(i - 1, b),
                        atol)
                << "valids propagation failed at i=" << i << ", b=" << b;
          }
        }
      }
    }
  });
}

TEST_F(ExtractOrderedTest, BasicCorrectness) {
  // 设置 MPC 环境参数：2方计算，SEMI2K 协议，64位环
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

        // -------------------------------------------------------
        // 1. 准备输入数据 - 支持多个数组
        // -------------------------------------------------------
        int64_t num_arrays = 2;
        int64_t n = 6;
        xt::xarray<int64_t> x = {{0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}};
        // valid bits
        xt::xarray<int64_t> valids = {0, 1, 0, 1, 1, 0};

        // 预期输出（提取有效元素）
        xt::xarray<int64_t> x_out_expected = {{1, 3, 4}, {11, 13, 14}};

        auto x_in = test::makeValue(&ctx, x, VIS_SECRET);
        valids.reshape({1, static_cast<size_t>(n)});
        auto valids_in = test::makeValue(&ctx, valids, VIS_SECRET);

        // -------------------------------------------------------
        // 2. 执行协议并记录性能指标
        // -------------------------------------------------------

        // 记录开始状态
        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // === 调用核心函数 ===
        auto res = extract_ordered(&ctx, x_in, valids_in);
        auto& y = res.first;
        auto valid_count = res.second;
        // ===================

        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();

        // -------------------------------------------------------
        // 3. 打印统计信息
        // -------------------------------------------------------
        stats = lctx->GetStats();  // 刷新统计
        size_t end_bytes = stats->sent_bytes;
        size_t end_actions = stats->sent_actions;

        if (lctx->Rank() == 0) {
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count();
          auto comm_bytes = end_bytes - start_bytes;
          auto comm_rounds = end_actions - start_actions;
          double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

          std::cout << "\n========================================"
                    << std::endl;
          std::cout << "ExtractOrdered Protocol Execution Stats:" << std::endl;
          std::cout << "  - Input Shape : "
                    << " num_arrays = " << num_arrays << ", n = " << n
                    << std::endl;
          std::cout << "  - Protocol    : " << protocol << std::endl;
          std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
          std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
                    << comm_mb << " MB)" << std::endl;
          std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                    << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        // -------------------------------------------------------
        // 4. 验证结果
        // -------------------------------------------------------
        EXPECT_EQ(valid_count, x_out_expected.shape()[1]);

        for (int64_t i = 0; i < num_arrays; ++i) {
          auto y_revealed = hal::reveal(&ctx, y[i]);
          auto y_row = hal::dump_public_as<int64_t>(&ctx, y_revealed);

          // 截取有效部分 [0, valid_count)
          // 注意：y_row 可能是 (1, n) 或 (n)，视具体实现而定，这里统一处理
          xt::xarray<int64_t> y_valid_part;
          if (y_row.dimension() == 2) {
            y_valid_part = xt::view(y_row, 0, xt::range(0, valid_count));
          } else {
            y_valid_part = xt::view(y_row, xt::range(0, valid_count));
          }
          auto y_expected_row = xt::row(x_out_expected, i);
          EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));

          if (lctx->Rank() == 0) {
            std::cout << "Array " << i << " Result (Valid): " << y_valid_part
                      << std::endl;
            std::cout << "Array " << i << " Expected      : " << y_expected_row
                      << std::endl
                      << std::endl;
          }
        }
      });
}

// 大规模输入测试：验证在 n=1024 情况下的正确性和稳定性
TEST_F(ExtractOrderedTest, LargeScale) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // -------------------------------------------------------
    // 1. 准备输入数据 - 支持多个数组
    // -------------------------------------------------------
    const int64_t num_arrays = 5;
    const int64_t n = 1000000;
    std::vector<std::vector<int64_t>> xs(num_arrays, std::vector<int64_t>(n));
    for (int64_t r = 0; r < num_arrays; ++r) {
      int64_t base = r * 100000;  // 不同行不同基数
      for (int64_t i = 0; i < n; ++i) {
        xs[r][i] = base + i;
      }
    }

    // valid bits
    std::vector<int64_t> valids(n);
    for (int64_t i = 0; i < n; ++i) {
      valids[i] = (i % 5 == 0) ? 1 : 0;
    }

    // 扁平化 xs 为 [num_arrays * n]，然后 reshape 为 [num_arrays, n]
    std::vector<int64_t> x_combined;
    x_combined.reserve(num_arrays * n);
    for (int64_t r = 0; r < num_arrays; ++r) {
      x_combined.insert(x_combined.end(), xs[r].begin(), xs[r].end());
    }
    xt::xarray<int64_t> x_arr = xt::adapt(x_combined);
    x_arr.reshape({static_cast<size_t>(num_arrays), static_cast<size_t>(n)});
    auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

    // valids 为 [1, n]
    xt::xarray<int64_t> f_arr = xt::adapt(valids);
    f_arr.reshape({1, static_cast<size_t>(n)});
    auto f_in = test::makeValue(&ctx, f_arr, VIS_SECRET);

    // 准备 Ground Truth
    std::vector<std::vector<int64_t>> exps(num_arrays);
    for (int64_t i = 0; i < n; ++i) {
      if (valids[i]) {
        for (int64_t r = 0; r < num_arrays; ++r) {
          exps[r].push_back(xs[r][i]);
        }
      }
    }
    size_t valid_count_expected = exps.empty() ? 0 : exps[0].size();

    // 扁平化预期并 reshape 为 [num_arrays, valid_count_expected]
    std::vector<int64_t> y_combined;
    y_combined.reserve(num_arrays * valid_count_expected);
    for (int64_t r = 0; r < num_arrays; ++r) {
      y_combined.insert(y_combined.end(), exps[r].begin(), exps[r].end());
    }
    xt::xarray<int64_t> x_out_expected = xt::adapt(y_combined);
    x_out_expected.reshape({static_cast<size_t>(num_arrays),
                            static_cast<size_t>(valid_count_expected)});

    // -------------------------------------------------------
    // 2. 执行协议并记录性能指标
    // -------------------------------------------------------
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto res = extract_ordered(&ctx, x_in, f_in);
    auto& y = res.first;
    auto valid_count = res.second;

    // -------------------------------------------------------
    // 3. 打印统计信息
    // -------------------------------------------------------
    auto end_time = std::chrono::high_resolution_clock::now();
    stats = lctx->GetStats();  // 刷新统计
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;
    if (lctx->Rank() == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();
      auto comm_bytes = end_bytes - start_bytes;
      auto comm_rounds = end_actions - start_actions;
      double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

      std::cout << "\n========================================" << std::endl;
      std::cout << "ExtractOrdered Protocol Execution Stats:" << std::endl;
      std::cout << "  - Input Shape : "
                << " num_arrays = " << num_arrays << ", n = " << n << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
      std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes (" << comm_mb
                << " MB)" << std::endl;
      std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    EXPECT_EQ(valid_count, static_cast<int64_t>(valid_count_expected));

    for (int64_t i = 0; i < num_arrays; ++i) {
      auto y_revealed = hal::reveal(&ctx, y[i]);
      auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

      xt::xarray<int64_t> y_valid_part;
      if (y_vec.dimension() == 2) {
        y_valid_part = xt::view(y_vec, 0, xt::range(0, valid_count));
      } else {
        y_valid_part = xt::view(y_vec, xt::range(0, valid_count));
      }

      auto y_expected_row = xt::row(x_out_expected, i);

      if (lctx->Rank() == 0) {
        std::cout << "LargeScale Array " << i << " valid_count=" << valid_count
                  << std::endl;
      }

      EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));
    }
  });
}

TEST(LogstarTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
        xt::xarray<float> x = {1, 3, 20};
        xt::xarray<float> y = {2, 50, 60};
        if (lctx->Rank() == 0) {
          std::cout << "x = \n" << x << std::endl;
          std::cout << "y = \n" << y << std::endl;
        }
        xt::xarray<float> res_expected = {1, 2, 3, 20, 50, 60};
        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        auto y_s = test::makeValue(&ctx, y, VIS_SECRET);
        setupTrace(&ctx, ctx.config());

        // Merge
        logstar(&ctx, x_s, y_s);

        // test::printProfileData(&ctx);
      });
}

}  // namespace spu::kernel::hal