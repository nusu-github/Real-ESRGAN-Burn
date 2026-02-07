#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Rrdb { scale: usize, num_block: usize },
    Srvgg { scale: usize, num_conv: usize },
    Discriminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    RealEsrganX4Plus,
    RealEsrganX2Plus,
    RealEsrnetX4Plus,
    EsrganSrX4Df2kostOfficial,
    RealEsrganX4PlusAnime6B,
    RealEsrAnimeVideoV3,
    RealEsrGeneralX4V3,
    RealEsrganX4PlusNetD,
    RealEsrganX2PlusNetD,
    RealEsrganX4PlusAnime6BNetD,
}

impl ModelVariant {
    pub const fn all() -> [Self; 10] {
        [
            Self::RealEsrganX4Plus,
            Self::RealEsrganX2Plus,
            Self::RealEsrnetX4Plus,
            Self::EsrganSrX4Df2kostOfficial,
            Self::RealEsrganX4PlusAnime6B,
            Self::RealEsrAnimeVideoV3,
            Self::RealEsrGeneralX4V3,
            Self::RealEsrganX4PlusNetD,
            Self::RealEsrganX2PlusNetD,
            Self::RealEsrganX4PlusAnime6BNetD,
        ]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::RealEsrganX4Plus => "RealESRGAN_x4plus",
            Self::RealEsrganX2Plus => "RealESRGAN_x2plus",
            Self::RealEsrnetX4Plus => "RealESRNet_x4plus",
            Self::EsrganSrX4Df2kostOfficial => "ESRGAN_SRx4_DF2KOST_official",
            Self::RealEsrganX4PlusAnime6B => "RealESRGAN_x4plus_anime_6B",
            Self::RealEsrAnimeVideoV3 => "realesr-animevideov3",
            Self::RealEsrGeneralX4V3 => "realesr-general-x4v3",
            Self::RealEsrganX4PlusNetD => "RealESRGAN_x4plus_netD",
            Self::RealEsrganX2PlusNetD => "RealESRGAN_x2plus_netD",
            Self::RealEsrganX4PlusAnime6BNetD => "RealESRGAN_x4plus_anime_6B_netD",
        }
    }

    pub const fn output_slug(self) -> &'static str {
        self.name()
    }

    pub const fn file_name(self) -> &'static str {
        match self {
            Self::RealEsrganX4Plus => "RealESRGAN_x4plus.pth",
            Self::RealEsrganX2Plus => "RealESRGAN_x2plus.pth",
            Self::RealEsrnetX4Plus => "RealESRNet_x4plus.pth",
            Self::EsrganSrX4Df2kostOfficial => "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
            Self::RealEsrganX4PlusAnime6B => "RealESRGAN_x4plus_anime_6B.pth",
            Self::RealEsrAnimeVideoV3 => "realesr-animevideov3.pth",
            Self::RealEsrGeneralX4V3 => "realesr-general-x4v3.pth",
            Self::RealEsrganX4PlusNetD => "RealESRGAN_x4plus_netD.pth",
            Self::RealEsrganX2PlusNetD => "RealESRGAN_x2plus_netD.pth",
            Self::RealEsrganX4PlusAnime6BNetD => "RealESRGAN_x4plus_anime_6B_netD.pth",
        }
    }

    pub const fn architecture(self) -> Architecture {
        match self {
            Self::RealEsrganX4Plus => Architecture::Rrdb {
                scale: 4,
                num_block: 23,
            },
            Self::RealEsrganX2Plus => Architecture::Rrdb {
                scale: 2,
                num_block: 23,
            },
            Self::RealEsrnetX4Plus => Architecture::Rrdb {
                scale: 4,
                num_block: 23,
            },
            Self::EsrganSrX4Df2kostOfficial => Architecture::Rrdb {
                scale: 4,
                num_block: 23,
            },
            Self::RealEsrganX4PlusAnime6B => Architecture::Rrdb {
                scale: 4,
                num_block: 6,
            },
            Self::RealEsrAnimeVideoV3 => Architecture::Srvgg {
                scale: 4,
                num_conv: 16,
            },
            Self::RealEsrGeneralX4V3 => Architecture::Srvgg {
                scale: 4,
                num_conv: 32,
            },
            Self::RealEsrganX4PlusNetD
            | Self::RealEsrganX2PlusNetD
            | Self::RealEsrganX4PlusAnime6BNetD => Architecture::Discriminator,
        }
    }

    pub const fn top_level_keys(self) -> &'static [&'static str] {
        match self {
            Self::RealEsrganX4Plus
            | Self::RealEsrganX2Plus
            | Self::RealEsrnetX4Plus
            | Self::RealEsrganX4PlusAnime6B => &["params_ema", "params"],
            Self::EsrganSrX4Df2kostOfficial
            | Self::RealEsrAnimeVideoV3
            | Self::RealEsrGeneralX4V3
            | Self::RealEsrganX4PlusNetD
            | Self::RealEsrganX2PlusNetD
            | Self::RealEsrganX4PlusAnime6BNetD => &["params"],
        }
    }

    pub const fn is_generator(self) -> bool {
        !matches!(self.architecture(), Architecture::Discriminator)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelVariant;

    #[test]
    fn all_variants_are_listed() {
        let all = ModelVariant::all();
        assert_eq!(all.len(), 10);
        assert_eq!(all[0].name(), "RealESRGAN_x4plus");
        assert_eq!(all[9].name(), "RealESRGAN_x4plus_anime_6B_netD");
    }
}
