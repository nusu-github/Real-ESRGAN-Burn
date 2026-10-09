//! The official Real-ESRGAN checkpoints and the networks they belong to.

use std::fmt;
use std::str::FromStr;

use crate::arch::{
    GeneratorConfig, RrdbNetConfig, SrvggNetCompactConfig, UnetDiscriminatorSnConfig,
};

#[derive(Debug, Clone)]
pub enum Architecture {
    Generator(GeneratorConfig),
    Discriminator(UnetDiscriminatorSnConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub const ALL: [Self; 10] = [
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
    ];

    /// The official model name, e.g. `RealESRGAN_x4plus`.
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

    /// File name of the official release asset.
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

    pub fn architecture(self) -> Architecture {
        let rrdb = |scale, num_block| {
            Architecture::Generator(GeneratorConfig::Rrdb(
                RrdbNetConfig::new(scale).with_num_block(num_block),
            ))
        };
        let srvgg = |num_conv| {
            Architecture::Generator(GeneratorConfig::Srvgg(
                SrvggNetCompactConfig::new(4).with_num_conv(num_conv),
            ))
        };

        match self {
            Self::RealEsrganX4Plus | Self::RealEsrnetX4Plus | Self::EsrganSrX4Df2kostOfficial => {
                rrdb(4, 23)
            }
            Self::RealEsrganX2Plus => rrdb(2, 23),
            Self::RealEsrganX4PlusAnime6B => rrdb(4, 6),
            Self::RealEsrAnimeVideoV3 => srvgg(16),
            Self::RealEsrGeneralX4V3 => srvgg(32),
            Self::RealEsrganX4PlusNetD
            | Self::RealEsrganX2PlusNetD
            | Self::RealEsrganX4PlusAnime6BNetD => {
                Architecture::Discriminator(UnetDiscriminatorSnConfig::new())
            }
        }
    }

    /// Keys under which the checkpoint may store its state dict, in order of preference.
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
}

impl fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownModel(pub String);

impl fmt::Display for UnknownModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = ModelVariant::ALL.iter().map(|v| v.name()).collect();
        write!(
            f,
            "unknown model `{}`; expected one of: {}",
            self.0,
            names.join(", ")
        )
    }
}

impl std::error::Error for UnknownModel {}

impl FromStr for ModelVariant {
    type Err = UnknownModel;

    /// Parses the official model name (see [`ModelVariant::name`]).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|variant| variant.name() == name)
            .ok_or_else(|| UnknownModel(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for variant in ModelVariant::ALL {
            assert_eq!(variant.name().parse::<ModelVariant>(), Ok(variant));
            assert!(variant.file_name().starts_with(variant.name()));
        }
        assert!("RealESRGAN_x8plus".parse::<ModelVariant>().is_err());
    }

    #[test]
    fn generator_scales() {
        let scales: Vec<_> = ModelVariant::ALL
            .into_iter()
            .filter_map(|variant| match variant.architecture() {
                Architecture::Generator(config) => Some(config.scale()),
                Architecture::Discriminator(_) => None,
            })
            .collect();
        assert_eq!(scales, [4, 2, 4, 4, 4, 4, 4]);
    }
}
